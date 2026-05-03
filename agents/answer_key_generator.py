"""
Agent 6 — Answer Key Generator

Produces exactly one answer-key entry per input question, even if the LLM returns
extra/missing rows (alignment + merge).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


class AnswerKeyGeneratorAgent:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, questions: List[Dict]) -> List[Dict]:
        if not questions:
            return []

        n = len(questions)
        prompt = f"""
You are an expert university examiner.

Generate a model answer and marking scheme for EACH of the {n} questions below.

Rules:
- Return a JSON array and NOTHING ELSE (no markdown fences, no commentary).
- The array MUST contain EXACTLY {n} objects.
- Object order MUST match the input question order (first object = first question, etc.).
- Each object must include: "question" (same text as input), "model_answer",
  "marking_scheme" (array of strings), "recommended_marks" (number).

Input questions (length {n}):
{questions}

Output format:
[
  {{
    "question": "...",
    "model_answer": "...",
    "marking_scheme": ["point 1", "point 2"],
    "recommended_marks": 5
  }}
]
"""
        response = self.llm.invoke(prompt)
        return self._safe_parse(response.content, questions)

    def _fallback_entry(self, q: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "question": q.get("question", ""),
            "model_answer": "Answer generation failed. Please regenerate.",
            "marking_scheme": ["Manual marking required."],
            "recommended_marks": 0,
        }

    def _merge_answer_entry(self, parsed: Dict[str, Any], q: Dict[str, Any]) -> Dict[str, Any]:
        entry = dict(parsed) if isinstance(parsed, dict) else {}
        entry["question"] = q.get("question", entry.get("question", ""))
        entry.setdefault("model_answer", "")
        ms = entry.get("marking_scheme", [])
        if isinstance(ms, str):
            entry["marking_scheme"] = [ms] if ms else []
        elif not isinstance(ms, list):
            entry["marking_scheme"] = []
        entry.setdefault("recommended_marks", 0)
        try:
            entry["recommended_marks"] = int(entry["recommended_marks"])
        except (TypeError, ValueError):
            entry["recommended_marks"] = 0
        return entry

    def _norm(self, s: str) -> str:
        return " ".join(str(s).lower().split())[:4000]

    def _align_answer_key(
        self, parsed: List[Dict[str, Any]], questions: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Ensure len(output) == len(questions); match by text overlap when counts differ."""
        n = len(questions)
        if n == 0:
            return []
        if not parsed:
            return [self._fallback_entry(q) for q in questions]

        used: set[int] = set()
        out: List[Dict[str, Any]] = []

        for i, q in enumerate(questions):
            qt = self._norm(q.get("question", ""))
            best_j: int | None = None
            best_score = -1.0

            for j, p in enumerate(parsed):
                if j in used:
                    continue
                pq = self._norm(p.get("question", ""))
                if not pq:
                    score = -1.0
                elif pq == qt:
                    score = 100.0
                elif qt and pq:
                    if qt in pq or pq in qt:
                        score = 85.0
                    else:
                        ts, ps = set(qt.split()), set(pq.split())
                        inter = len(ts & ps)
                        score = 70.0 * inter / max(min(len(ts), len(ps)), 1)
                else:
                    score = -1.0

                if score > best_score:
                    best_score = score
                    best_j = j

            chosen: Dict[str, Any] | None = None
            if best_j is not None and best_score >= 25.0:
                used.add(best_j)
                chosen = parsed[best_j]
            elif i < len(parsed) and i not in used:
                used.add(i)
                chosen = parsed[i]
            else:
                for j in range(len(parsed)):
                    if j not in used:
                        used.add(j)
                        chosen = parsed[j]
                        break

            if chosen is not None:
                out.append(self._merge_answer_entry(chosen, q))
            else:
                out.append(self._fallback_entry(q))

        return out[:n]

    def _extract_json_array(self, text: str) -> List[Any] | None:
        text = text.strip()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        try:
            match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return data
        except json.JSONDecodeError:
            pass
        return None

    def _safe_parse(self, text: str, questions: List[Dict]) -> List[Dict]:
        parsed_raw = self._extract_json_array(text)
        if parsed_raw is None:
            return [self._fallback_entry(q) for q in questions]

        parsed: List[Dict[str, Any]] = []
        for item in parsed_raw:
            if isinstance(item, dict):
                parsed.append(item)

        n = len(questions)
        if len(parsed) == n:
            aligned = [self._merge_answer_entry(parsed[i], questions[i]) for i in range(n)]
            return aligned

        # Wrong count: greedy align by question text, then positional fill.
        return self._align_answer_key(parsed, questions)
