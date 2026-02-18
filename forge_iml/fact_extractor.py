"""
IML Fact Extractor -- NER-D pipeline for atomic fact extraction.

Extracts (subject, predicate, object) triples from conversations,
deduplicates against a MemoryProvider, and stores new facts.
Runs asynchronously after response delivery -- never blocks chat.
"""

import asyncio
import json
import logging
import re
import time
from typing import Optional

from forge_iml.providers.base import LLMProvider, MemoryProvider

log = logging.getLogger("forge_iml.fact_extractor")

# -- Config -------------------------------------------------------------------

MIN_MESSAGE_LENGTH = 10
RATE_LIMIT_SECONDS = 30

# -- Extraction Prompt --------------------------------------------------------

EXTRACTION_PROMPT = """\
Extract all factual claims from this conversation between a user and an AI assistant.
Return ONLY valid JSON -- no markdown, no explanation.

Rules:
- Only extract concrete facts, not opinions or greetings
- Each fact must be an atomic (subject, predicate, object) triple
- Confidence: 0.9 for explicit statements, 0.7 for implied, 0.5 for uncertain
- Skip questions unless they contain embedded facts
- Subject should be a named entity (person, place, thing)
- Predicate should be a verb or relationship
- Object should be specific (not vague)

Conversation:
User: {user_message}
Assistant: {assistant_response}

Return format:
{{"facts": [{{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.9}}]}}
If no facts found, return: {{"facts": []}}"""


# -- Main Class ---------------------------------------------------------------

class IMLFactExtractor:
    """Extracts atomic facts from conversations and stores them.

    Args:
        llm: LLMProvider for fact extraction.
        memory_provider: Optional MemoryProvider for dedup and storage.
    """

    def __init__(
        self,
        llm: Optional[LLMProvider] = None,
        memory_provider: Optional[MemoryProvider] = None,
    ):
        self._llm = llm
        self._memory = memory_provider
        self._last_extraction: dict[str, float] = {}

    # -- Public API -----------------------------------------------------------

    async def extract_and_store(
        self,
        user_message: str,
        assistant_response: str,
        user_handle: str,
    ) -> list[dict]:
        """Extract facts from a conversation turn and store new ones.

        Returns list of stored facts. Designed to be fire-and-forget.
        """
        if len(user_message.strip()) < MIN_MESSAGE_LENGTH:
            return []
        if self._is_question_only(user_message):
            return []

        now = time.time()
        last = self._last_extraction.get(user_handle, 0)
        if now - last < RATE_LIMIT_SECONDS:
            return []
        self._last_extraction[user_handle] = now

        raw_facts = await self._extract_facts(user_message, assistant_response)
        if not raw_facts:
            return []

        new_facts = await self._deduplicate(raw_facts)
        if not new_facts:
            return []

        stored = await self._store_facts(new_facts, user_handle)
        if stored:
            log.info("Extracted and stored %d facts from %s", len(stored), user_handle)
        return stored

    # -- Extraction -----------------------------------------------------------

    async def _extract_facts(self, user_message: str,
                              assistant_response: str) -> list[dict]:
        if self._llm is None:
            log.warning("No LLM provider -- cannot extract facts")
            return []

        prompt = EXTRACTION_PROMPT.format(
            user_message=user_message[:1000],
            assistant_response=assistant_response[:1000],
        )

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._llm.complete(
                    [{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1024,
                ),
            )
        except Exception as e:
            log.warning("LLM fact extraction call failed: %s", e)
            return []

        if not result or not result.get("response"):
            return []

        return self._parse_facts_json(result["response"])

    def _parse_facts_json(self, text: str) -> list[dict]:
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return []
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return []

        facts = data.get("facts", [])
        if not isinstance(facts, list):
            return []

        valid: list[dict] = []
        for f in facts:
            if not isinstance(f, dict):
                continue
            subj = str(f.get("subject", "")).strip()
            pred = str(f.get("predicate", "")).strip()
            obj = str(f.get("object", "")).strip()
            conf = float(f.get("confidence", 0.7))
            if subj and pred and obj:
                valid.append({
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
                    "confidence": min(max(conf, 0.0), 1.0),
                })
        return valid

    # -- Deduplication --------------------------------------------------------

    async def _deduplicate(self, facts: list[dict]) -> list[dict]:
        if self._memory is None:
            return facts

        new_facts: list[dict] = []
        for fact in facts:
            query = f"{fact[subject]} {fact[predicate]}"
            try:
                existing = self._memory.search(query, limit=3)
                is_duplicate = False
                update_id = None
                for mem in existing:
                    content = mem.get("content", "").lower()
                    s_match = fact["subject"].lower() in content
                    p_match = fact["predicate"].lower() in content
                    o_match = fact["object"].lower() in content

                    if s_match and p_match and o_match:
                        is_duplicate = True
                        break
                    elif s_match and p_match and not o_match:
                        update_id = mem.get("id")

                if is_duplicate:
                    continue
                if update_id:
                    fact["_update_id"] = update_id
                new_facts.append(fact)
            except Exception as e:
                log.debug("Dedup check failed: %s", e)
                new_facts.append(fact)

        return new_facts

    # -- Storage --------------------------------------------------------------

    async def _store_facts(self, facts: list[dict],
                            user_handle: str) -> list[dict]:
        if self._memory is None:
            return []

        stored: list[dict] = []
        for fact in facts:
            content = f"{fact[subject]} {fact[predicate]} {fact[object]}"
            tags = [fact["subject"].lower(), fact["predicate"].lower(),
                    user_handle.lower()]

            try:
                update_id = fact.pop("_update_id", None)
                if update_id:
                    ok = self._memory.update(update_id, content, tags=tags,
                                             importance=5)
                    if ok:
                        stored.append(fact)
                    continue

                entry_id = self._memory.save(
                    content,
                    namespace="facts",
                    importance=5,
                    tags=tags,
                    agent="forge-iml-facts",
                    entry_type="fact",
                )
                if entry_id:
                    stored.append(fact)
            except Exception as e:
                log.warning("Error storing fact: %s", e)

        return stored

    # -- Helpers --------------------------------------------------------------

    def _is_question_only(self, text: str) -> bool:
        text = text.strip()
        if len(text) < 50 and text.endswith("?"):
            return True
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return True
        question_words = {"who", "what", "where", "when", "why", "how", "is", "are",
                          "do", "does", "did", "can", "could", "would", "should", "will"}
        all_questions = all(
            s.split()[0].lower() in question_words if s.split() else True
            for s in sentences
        )
        return all_questions and len(text) < 200
