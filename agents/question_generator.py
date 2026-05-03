from typing import List, Dict
from rag_pipeline import retrieve_context, COLLECTION_SYLLABUS, COLLECTION_PAST_PAPERS
import json
import re


class QuestionGeneratorAgent:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings

    def generate_questions(self, topic: str, num_questions: int = 10) -> List[Dict]:
        """
        Generate questions for a given topic using RAG context.
        Returns structured questions.
        """

        # --- Retrieve context ---
        syllabus_context = retrieve_context(
            topic,
            COLLECTION_SYLLABUS,
            self.embeddings,
            k=5
        )

        past_context = retrieve_context(
            topic,
            COLLECTION_PAST_PAPERS,
            self.embeddings,
            k=3
        )

        # --- Prompt ---
        prompt = f"""
You are an expert university exam setter.

Generate {num_questions} high-quality exam questions.

Topic: {topic}

Use:
1. Course content:
{syllabus_context}

2. Past exam style:
{past_context}

Requirements:
- Include MCQs, short, and long questions
- Avoid duplication
- Maintain academic quality
- Cover different difficulty levels

Output format (STRICT JSON):
[
  {{
    "question": "...",
    "type": "MCQ | SHORT | LONG"
  }}
]
"""

        response = self.llm.invoke(prompt)
        return self._parse_response(response.content)

    # if not isinstance(parsed_output, list):
    # print("⚠️ Output not structured properly")

    def _parse_response(self, text: str):
        # Attempt 1: direct parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return self._normalize(parsed)
        except Exception:
            pass

        # Attempt 2: recover JSON array from model chatter
        try:
            match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return self._normalize(parsed)
        except Exception:
            pass

        print("[WARN] QuestionGenerator: failed to parse JSON. Returning fallback.")
        return [{"question": text.strip(), "type": "UNKNOWN"}]

    def _normalize(self, questions: List[Dict]) -> List[Dict]:
        valid_types = {"MCQ", "SHORT", "LONG"}
        normalized = []
        for q in questions:
            q_type = str(q.get("type", "UNKNOWN")).strip().upper()
            if q_type not in valid_types:
                q_type = "UNKNOWN"
            normalized.append({
                "question": str(q.get("question", "")).strip(),
                "type": q_type,
            })
        return [q for q in normalized if q["question"]]
