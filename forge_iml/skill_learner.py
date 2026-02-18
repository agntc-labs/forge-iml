"""
IML Skill Learner -- Pattern-based skill template learning.

Tracks successful execution patterns. When a pattern appears 3+ times,
promotes it as a skill template that can bypass full IML parsing.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

from forge_iml.schema import (
    IntentGraph,
    Node,
    NodeType,
    EdgeRelation,
    Source,
    Budget,
)

log = logging.getLogger("forge_iml.skill_learner")

# -- Config -------------------------------------------------------------------

PROMOTION_THRESHOLD = 3
MAX_TRACKED_PATTERNS = 500


# -- Helpers ------------------------------------------------------------------

def _extract_signature(graph: IntentGraph) -> Optional[str]:
    """Build a canonical pattern signature from a completed graph."""
    goals = graph.get_goals()
    if not goals:
        return None

    goal = goals[0]
    verb = goal.metadata.get("verb", "").lower().strip()
    obj = goal.metadata.get("object", "").lower().strip()

    if not verb and not obj:
        words = goal.content.lower().split()
        if len(words) >= 2:
            verb = words[0]
            obj = " ".join(words[1:3])
        else:
            return None

    steps = graph.get_plan_steps()
    tools: list[str] = []
    for step in steps:
        tool = step.metadata.get("tool", "")
        if tool and tool not in tools:
            tools.append(tool)

    if not tools:
        return None

    tool_str = ",".join(sorted(tools))
    return f"{verb}|{obj}|{tool_str}"


def _normalize_verb(text: str) -> str:
    synonyms = {
        "check": ["look up", "find out", "tell me", "show me", "get"],
        "get": ["fetch", "retrieve", "pull", "grab"],
        "send": ["deliver", "forward", "transmit"],
        "create": ["make", "build", "generate", "write"],
        "delete": ["remove", "erase", "drop", "clear"],
        "update": ["change", "modify", "edit", "set"],
    }
    text_lower = text.lower().strip()
    for canonical, alts in synonyms.items():
        if text_lower in alts:
            return canonical
    return text_lower


# -- Main Class ---------------------------------------------------------------

class IMLSkillLearner:
    """Learns from successful executions.

    After ``promotion_threshold`` identical patterns, promotes the pattern
    as a reusable skill template.

    Args:
        skills_dir: Directory for storing learned/promoted skills.
        promotion_threshold: Number of successes before promotion.
    """

    def __init__(
        self,
        skills_dir: Optional[Path] = None,
        promotion_threshold: int = PROMOTION_THRESHOLD,
    ):
        self._skills_dir = skills_dir or Path("iml_skills")
        self._learned_file = self._skills_dir / "iml_learned.json"
        self._promoted_dir = self._skills_dir / "iml_promoted"
        self._promotion_threshold = promotion_threshold
        self._ensure_dirs()
        self._patterns: dict = self._load_patterns()

    # -- Public API -----------------------------------------------------------

    def record_success(self, graph: IntentGraph) -> Optional[str]:
        """Record a successful execution pattern. Returns signature if promoted."""
        sig = _extract_signature(graph)
        if sig is None:
            return None

        patterns = self._load_patterns()

        if sig in patterns:
            entry = patterns[sig]
            entry["count"] += 1
            entry["last_seen"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        else:
            if len(patterns) >= MAX_TRACKED_PATTERNS:
                self._evict_oldest(patterns)
            entry = {
                "count": 1,
                "last_seen": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "template": graph.to_dict(),
                "promoted": False,
            }
            patterns[sig] = entry

        promoted_sig = None
        if entry["count"] >= self._promotion_threshold and not entry.get("promoted"):
            self._promote_skill(sig, graph)
            entry["promoted"] = True
            promoted_sig = sig
            log.info("Promoted skill: %s (count=%d)", sig, entry["count"])

        self._save_patterns(patterns)
        self._patterns = patterns
        return promoted_sig

    def get_skill_template(self, raw_input: str) -> Optional[IntentGraph]:
        """Check if an incoming request matches a promoted skill."""
        if not raw_input or not raw_input.strip():
            return None

        words = raw_input.lower().strip().split()
        if len(words) < 2:
            return None

        verb = _normalize_verb(words[0])
        obj_words = " ".join(words[1:4]).lower()

        promoted_files = list(self._promoted_dir.glob("*.json")) if self._promoted_dir.exists() else []
        best_match: Optional[tuple[str, dict]] = None
        best_score = 0

        for skill_file in promoted_files:
            try:
                with open(skill_file, "r") as f:
                    skill_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            sig = skill_data.get("signature", "")
            parts = sig.split("|")
            if len(parts) < 3:
                continue

            skill_verb, skill_obj = parts[0], parts[1]

            score = 0
            if verb == skill_verb or _normalize_verb(verb) == _normalize_verb(skill_verb):
                score += 2
            if skill_obj in obj_words or obj_words in skill_obj:
                score += 3
            skill_obj_words = set(skill_obj.split())
            input_obj_words = set(obj_words.split())
            overlap = skill_obj_words & input_obj_words
            score += len(overlap)

            if score > best_score and score >= 3:
                best_score = score
                best_match = (sig, skill_data)

        if best_match is None:
            return None

        sig, skill_data = best_match
        template_dict = skill_data.get("template", {})
        if not template_dict:
            return None

        try:
            graph = IntentGraph.from_dict(template_dict)
        except Exception as e:
            log.warning("Failed to deserialize skill template %s: %s", sig, e)
            return None

        graph = self._fill_variables(graph, raw_input)
        graph.source.raw_input = raw_input
        graph.cache_key = f"skill:{sig}"

        log.info("Matched skill template: %s (score=%d)", sig, best_score)
        return graph

    def list_skills(self) -> list[dict]:
        """List all tracked patterns and their status."""
        patterns = self._load_patterns()
        skills: list[dict] = []
        for sig, entry in patterns.items():
            skills.append({
                "signature": sig,
                "count": entry.get("count", 0),
                "promoted": entry.get("promoted", False),
                "last_seen": entry.get("last_seen", ""),
            })
        skills.sort(key=lambda s: s["count"], reverse=True)
        return skills

    # -- Internal: Promotion --------------------------------------------------

    def _promote_skill(self, sig: str, graph: IntentGraph):
        self._promoted_dir.mkdir(parents=True, exist_ok=True)
        template = graph.to_dict()

        for node in template.get("nodes", []):
            if node.get("type") == NodeType.USER_GOAL.value:
                meta = node.get("metadata", {})
                for key in ("location", "recipient", "amount", "query"):
                    if key in meta:
                        meta[key] = "{" + key + "}"

        safe_name = re.sub(r"[^a-z0-9_]", "_", sig.lower())[:80]
        skill_path = self._promoted_dir / f"{safe_name}.json"

        skill_data = {
            "signature": sig,
            "template": template,
            "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "version": "1.0",
        }

        self._atomic_write(skill_path, skill_data)

    def _fill_variables(self, graph: IntentGraph, raw_input: str) -> IntentGraph:
        words = raw_input.strip().split()
        context: dict[str, str] = {}

        for i, word in enumerate(words):
            if word.lower() == "in" and i + 1 < len(words):
                context["location"] = " ".join(words[i + 1 : i + 3])
            elif word.lower() == "to" and i + 1 < len(words):
                context["recipient"] = " ".join(words[i + 1 : i + 3])
            elif word.lower() == "for" and i + 1 < len(words):
                context["amount"] = words[i + 1]

        if "query" not in context and len(words) > 2:
            context["query"] = " ".join(words[1:])

        for node in graph.nodes:
            if node.type == NodeType.USER_GOAL:
                for key, val in context.items():
                    placeholder = "{" + key + "}"
                    if placeholder in node.content:
                        node.content = node.content.replace(placeholder, val)
                    for mk, mv in list(node.metadata.items()):
                        if isinstance(mv, str) and placeholder in mv:
                            node.metadata[mk] = mv.replace(placeholder, val)

        return graph

    # -- Internal: Pattern storage --------------------------------------------

    def _ensure_dirs(self):
        self._skills_dir.mkdir(parents=True, exist_ok=True)
        self._promoted_dir.mkdir(parents=True, exist_ok=True)

    def _load_patterns(self) -> dict:
        if not self._learned_file.exists():
            return {}
        try:
            with open(self._learned_file, "r") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, IOError) as e:
            log.warning("Failed to load patterns: %s", e)
            return {}

    def _save_patterns(self, patterns: dict):
        self._atomic_write(self._learned_file, patterns)

    def _atomic_write(self, path: Path, data: dict):
        tmp_path = path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp_path), str(path))
        except Exception as e:
            log.warning("Failed to write %s: %s", path, e)
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def _evict_oldest(self, patterns: dict):
        if len(patterns) < MAX_TRACKED_PATTERNS:
            return
        candidates = [
            (sig, entry) for sig, entry in patterns.items()
            if not entry.get("promoted")
        ]
        candidates.sort(key=lambda x: x[1].get("last_seen", ""))
        to_remove = max(1, len(candidates) // 10)
        for sig, _ in candidates[:to_remove]:
            del patterns[sig]
