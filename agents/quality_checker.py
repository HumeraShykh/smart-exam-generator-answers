"""
Agent 5 — Quality Checker

Responsibilities:
1) Split accidental "mega-questions" where the model dumped a JSON array into one string
2) Remove empty/malformed questions
3) Remove exact duplicate questions
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import numpy as np

from rag_pipeline import retrieve_context, COLLECTION_SYLLABUS


class QualityCheckerAgent:
    def __init__(self, embeddings, duplicate_threshold: float = 0.9):
        self.embeddings = embeddings
        self.duplicate_threshold = duplicate_threshold

    def run(self, questions: List[Dict]) -> List[Dict]:
        expanded = self._split_embedded_json_questions(questions)
        filtered = self._drop_invalid(expanded)
        filtered = self._drop_exact_duplicates(filtered)
        return filtered

    def _split_embedded_json_questions(self, questions: List[Dict]) -> List[Dict]:
        """
        If A1/A4 returned one dict whose `question` field is actually a JSON array
        of questions, expand into separate dicts so downstream agents see real items.
        """
        out: List[Dict] = []
        for parent in questions:
            text = str(parent.get("question", "")).strip()
            if not text:
                continue
            parsed_list = self._try_parse_question_array(text)
            if parsed_list is None:
                out.append(parent)
                continue
            for item in parsed_list:
                if not isinstance(item, dict):
                    continue
                qtext = str(item.get("question", "")).strip()
                if not qtext:
                    continue
                merged = self._merge_child_question(parent, item, qtext)
                out.append(merged)
        return out

    def _try_parse_question_array(self, text: str) -> list | None:
        """Return list of dicts if `text` is a JSON array of question objects; else None."""
        if not text.startswith("["):
            return None
        if '"question"' not in text and "'question'" not in text:
            return None
        try:
            data = json.loads(text)
            if not isinstance(data, list) or not data:
                return None
            if not all(isinstance(x, dict) for x in data):
                return None
            if not any("question" in x for x in data):
                return None
            return data
        except (json.JSONDecodeError, TypeError):
            pass
        # Recover first [...] array via regex (some models add whitespace/noise)
        try:
            match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group(0))
            if isinstance(data, list) and data and all(isinstance(x, dict) for x in data):
                return data
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    def _merge_child_question(
        self, parent: Dict[str, Any], item: Dict[str, Any], qtext: str
    ) -> Dict[str, Any]:
        child = dict(parent)
        child["question"] = qtext
        for key in ("type", "blooms_level", "difficulty", "clo", "plo"):
            if item.get(key) not in (None, "", "UNMAPPED"):
                child[key] = item[key]
            elif parent.get(key) is not None:
                child[key] = parent.get(key)
        if item.get("type"):
            child["type"] = str(item["type"]).strip().upper()
        return child

    def _drop_invalid(self, questions: List[Dict]) -> List[Dict]:
        cleaned = []
        for q in questions:
            text = str(q.get("question", "")).strip()
            if not text:
                continue
            # Drop remaining garbage blobs that are still array-shaped but huge / unusable
            if text.startswith("[") and len(text) > 8000:
                continue
            cleaned.append(q)
        return cleaned

    def _drop_exact_duplicates(self, questions: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for q in questions:
            key = str(q["question"]).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(q)
        return unique

    def _drop_semantic_duplicates(self, questions: List[Dict]) -> List[Dict]:
        if len(questions) < 2:
            return questions

        texts = [q["question"] for q in questions]
        vectors = self.embeddings.embed_documents(texts)
        keep = []

        for idx, q in enumerate(questions):
            is_dup = False
            for kept_idx in keep:
                sim = self._cosine_similarity(vectors[idx], vectors[kept_idx])
                if sim >= self.duplicate_threshold:
                    is_dup = True
                    break
            if not is_dup:
                keep.append(idx)

        return [questions[i] for i in keep]

    def _cosine_similarity(self, a, b) -> float:
        a = np.array(a)
        b = np.array(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def check_grounding(self, question_text: str) -> str:
        return retrieve_context(question_text, COLLECTION_SYLLABUS, self.embeddings, k=2)
