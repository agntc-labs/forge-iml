"""
IML Policy -- Compiled safety gates.

Deterministic Python checks that run BEFORE tool execution.
The LLM cannot bypass these -- they are hard-coded policy.

All gates are evaluated in order; first triggered gate
short-circuits and blocks/requires confirmation.
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from forge_iml.schema import Node, NodeType, PolicyAction


# -- PolicyGate dataclass -----------------------------------------------------

@dataclass
class PolicyGate:
    name: str
    description: str
    blocks: list[str]              # tool names, or ["*"] for all
    condition: Callable[[Node, dict], bool]
    action: PolicyAction
    message: str                   # may contain {tool}, {detail}, {tier}


# -- Helpers ------------------------------------------------------------------

def _inputs_str(step: Node) -> str:
    """Stringify all inputs from a PlanStep metadata for pattern matching."""
    meta = step.metadata or {}
    inputs = meta.get("inputs", {})
    if isinstance(inputs, dict):
        return " ".join(str(v) for v in inputs.values()).lower()
    return str(inputs).lower()


def _tool_name(step: Node) -> str:
    """Extract tool name from a PlanStep node."""
    return (step.metadata or {}).get("tool", "")


def _matches_block_list(tool: str, blocks: list[str]) -> bool:
    """Check if a tool matches a gate block list."""
    if "*" in blocks:
        return True
    return tool in blocks


# -- Gate condition functions -------------------------------------------------

_SECRET_PATTERNS = [".env", "credentials", "secret", "token", "key", "password", ".pem", ".key"]

def _cond_no_secret_reads(step: Node, context: dict) -> bool:
    tool = _tool_name(step)
    if tool == "keychain_get":
        return True
    if tool == "read_file":
        inputs = _inputs_str(step)
        return any(pat in inputs for pat in _SECRET_PATTERNS)
    return False


def _cond_no_dotenv(step: Node, context: dict) -> bool:
    return ".env" in _inputs_str(step)


def _cond_no_wallet_keys(step: Node, context: dict) -> bool:
    wallet_pats = ["wallet", "private_key", "seed_phrase", "mnemonic"]
    inputs = _inputs_str(step)
    return any(pat in inputs for pat in wallet_pats)


def _cond_no_remote_install(step: Node, context: dict) -> bool:
    install_pats = [
        "curl | sh", "curl | bash", "wget | sh", "curl|sh", "curl|bash",
        "pip install --", "npm install -g", "brew install",
    ]
    inputs = _inputs_str(step)
    if any(pat in inputs for pat in install_pats):
        return True
    if re.search(r"curl\s+\S+.*\|\s*(sh|bash)", inputs):
        return True
    if re.search(r"wget\s+\S+.*\|\s*(sh|bash)", inputs):
        return True
    return False


_DESTRUCTIVE_PATTERNS = [
    "rm -rf", "drop table", "delete from", "git push --force",
    "git reset --hard", "format", "fdisk", "truncate",
]

def _cond_no_destructive(step: Node, context: dict) -> bool:
    inputs = _inputs_str(step)
    return any(pat in inputs for pat in _DESTRUCTIVE_PATTERNS)


def _cond_tool_tier_check(step: Node, context: dict) -> bool:
    ctx = context or {}
    allowed = ctx.get("user_allowed_tools", None)
    if allowed is None:
        return False
    tool = _tool_name(step)
    return tool not in allowed


# -- PolicyEngine -------------------------------------------------------------

class PolicyEngine:
    """Evaluates compiled safety gates against PlanStep nodes.

    The default gates cover common safety patterns (secret reads,
    destructive operations, remote installs). Override or extend by
    passing custom gates to ``__init__``.

    Args:
        gates: Optional list of PolicyGate instances. If not provided
            the default set is used.
    """

    def __init__(self, gates: Optional[list[PolicyGate]] = None):
        self._gates: list[PolicyGate] = gates if gates is not None else self._default_gates()
        self._checks_total = 0
        self._triggers_total = 0

    @staticmethod
    def _default_gates() -> list[PolicyGate]:
        return [
            PolicyGate(
                name="no_secret_reads",
                description="Block reading secrets, credentials, keys, or .env files",
                blocks=["keychain_get", "read_file"],
                condition=_cond_no_secret_reads,
                action=PolicyAction.BLOCK,
                message="Blocked: secret/credential access requires explicit user approval",
            ),
            PolicyGate(
                name="no_dotenv",
                description="Block any access to .env files",
                blocks=["read_file", "edit_file", "bash"],
                condition=_cond_no_dotenv,
                action=PolicyAction.BLOCK,
                message="Blocked: .env file access prohibited",
            ),
            PolicyGate(
                name="no_wallet_keys",
                description="Block access to wallet/crypto key material",
                blocks=["read_file", "bash"],
                condition=_cond_no_wallet_keys,
                action=PolicyAction.BLOCK,
                message="Blocked: wallet/crypto key access prohibited",
            ),
            PolicyGate(
                name="no_destructive_without_approval",
                description="Block destructive operations (rm -rf, drop table, force push, etc.)",
                blocks=["bash", "delete_file", "git_push"],
                condition=_cond_no_destructive,
                action=PolicyAction.BLOCK,
                message="Blocked: destructive operation requires explicit user approval",
            ),
            PolicyGate(
                name="no_remote_install",
                description="Require confirmation for remote package installs",
                blocks=["bash"],
                condition=_cond_no_remote_install,
                action=PolicyAction.REQUIRE_CONFIRMATION,
                message="Remote package install detected: {detail}. Approve?",
            ),
            PolicyGate(
                name="tool_tier_check",
                description="Block tools not in user allowed set",
                blocks=["*"],
                condition=_cond_tool_tier_check,
                action=PolicyAction.BLOCK,
                message="Blocked: tool {tool} not available for tier {tier}",
            ),
        ]

    def check_step(self, step: Node, context: dict = None) -> tuple[bool, str, str]:
        """Check a PlanStep node against all policy gates.

        Returns:
            (allowed, gate_name, message)
        """
        self._checks_total += 1
        ctx = context or {}
        tool = _tool_name(step)

        for gate in self._gates:
            if not _matches_block_list(tool, gate.blocks):
                continue
            if gate.condition(step, ctx):
                detail = _inputs_str(step)[:120]
                msg = gate.message.replace("{tool}", tool)
                msg = msg.replace("{detail}", detail)
                msg = msg.replace("{tier}", ctx.get("recipient_tier", "unknown"))
                allowed = gate.action == PolicyAction.ALLOW
                if not allowed:
                    self._triggers_total += 1
                return (allowed, gate.name, msg)

        return (True, "", "")

    def list_gates(self) -> list[dict]:
        """Return gate names and descriptions for transparency."""
        return [
            {
                "name": g.name,
                "description": g.description,
                "blocks": g.blocks,
                "action": g.action.value,
            }
            for g in self._gates
        ]

    def check_all(self, steps: list[Node], context: dict = None) -> list[tuple]:
        """Check multiple steps and return results for each."""
        return [self.check_step(step, context) for step in steps]

    @property
    def stats(self) -> dict:
        return {
            "checks_total": self._checks_total,
            "triggers_total": self._triggers_total,
        }
