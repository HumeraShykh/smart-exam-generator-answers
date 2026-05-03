"""
Multi-Agent Exam Orchestrator
6 agents run sequentially; CrewAI-style roles are preserved as class-level
metadata but execution is direct Python so no cloud LLM dependency is needed.

Agent pipeline:
  A1 QuestionGenerator  -> A2 BloomClassifier -> A3 DifficultyBalancer
  -> A4 CLOMapper (gap-fill loop) -> A5 QualityChecker -> A6 AnswerKeyGenerator
"""
from typing import Dict, List


from agents.answer_key_generator import AnswerKeyGeneratorAgent
from agents.bloom_classifier import BloomClassifierAgent
from agents.clo_mapper import CLOMapperAgent
from agents.difficulty_balancer import DifficultyBalancerAgent
from agents.quality_checker import QualityCheckerAgent
from agents.question_generator import QuestionGeneratorAgent


def _log(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode())


# ---------------------------------------------------------------------------
# Agent role descriptors (CrewAI-style metadata, used for logging/docs)
# ---------------------------------------------------------------------------

AGENT_ROLES = {
    "A1": {
        "role": "Exam Question Generator",
        "goal": "Generate diverse, high-quality exam questions covering all CLOs",
        "backstory": "Expert NLP university exam setter using RAG over syllabus and past papers.",
    },
    "A2": {
        "role": "Bloom's Taxonomy Classifier",
        "goal": "Classify each question into the correct Bloom's cognitive level",
        "backstory": "Education psychologist specialising in Bloom's revised taxonomy.",
    },
    "A3": {
        "role": "Difficulty Balancer",
        "goal": "Ensure 30% Easy / 40% Medium / 30% Hard question distribution",
        "backstory": "Assessment designer focused on balanced, fair exams.",
    },
    "A4": {
        "role": "CLO/PLO Mapper",
        "goal": "Map every question to a CLO and PLO using semantic similarity",
        "backstory": "OBE specialist who aligns assessments with learning outcomes.",
    },
    "A5": {
        "role": "Quality Checker",
        "goal": "Remove duplicate and low-quality questions",
        "backstory": "Senior examiner with a sharp eye for quality and originality.",
    },
    "A6": {
        "role": "Answer Key Generator",
        "goal": "Generate model answers and marking schemes for all questions",
        "backstory": "Academic with expertise in writing comprehensive answer keys.",
    },
}


def _agent_header(code: str) -> None:
    meta = AGENT_ROLES[code]
    _log(f"\n[{code}] {meta['role']}")
    _log(f"     Goal: {meta['goal']}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ExamOrchestrator:
    def __init__(self, pipeline: Dict):
        self.llm = pipeline["llm"]
        self.embeddings = pipeline["embeddings"]

        self._qg = QuestionGeneratorAgent(self.llm, self.embeddings)
        self._bc = BloomClassifierAgent(self.llm)
        self._db = DifficultyBalancerAgent()
        self._cm = CLOMapperAgent(self.embeddings)
        self._qc = QualityCheckerAgent(self.embeddings)
        self._ak = AnswerKeyGeneratorAgent(self.llm)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full(
        self,
        topic: str,
        max_iterations: int = 3,
        *,
        initial_questions: int = 12,
        on_step=None,
    ) -> Dict:
        """
        on_step(code, event, data) is called at each agent milestone so the
        UI can display live progress.
          code  : "A1".."A6"
          event : "start" | "done" | "iteration" (A4)
          data  : dict with agent-specific payload
        initial_questions: target size for the first A1 call; A4 may add more for CLO coverage.
        """
        def notify(code, event, **data):
            _agent_header(code)
            if on_step:
                try:
                    on_step(code, event, data)
                except Exception:
                    pass

        _log("\n=== MULTI-AGENT ORCHESTRATION STARTING ===")

        # A1 — Question Generation (batch size; A4 may still append gap-fill items)
        notify("A1", "start")
        n0 = max(1, min(int(initial_questions), 40))
        questions = self._qg.generate_questions(topic, num_questions=n0)
        _log(f"     Generated {len(questions)} questions.")
        notify("A1", "done", questions=questions)

        # A2 — Bloom Classification
        notify("A2", "start")
        questions = self._bc.classify(questions)
        _log(f"     Classified {len(questions)} questions.")
        notify("A2", "done", questions=questions)

        # A3 — Difficulty Balancing
        notify("A3", "start")
        questions = self._db.balance(questions)
        _log(f"     Balanced to {len(questions)} questions.")
        notify("A3", "done", questions=questions)

        # A4 — CLO/PLO Mapping
        notify("A4", "start")
        questions = self._run_clo_mapping(questions, max_iterations, on_step)
        notify("A4", "done", questions=questions)

        # A5 — Quality Check
        notify("A5", "start")
        questions = self._qc.run(questions)
        _log(f"     {len(questions)} questions passed quality check.")
        notify("A5", "done", questions=questions)

        # A6 — Answer Key
        notify("A6", "start")
        answer_key = self._ak.generate(questions)
        _log(f"     Answer key generated for {len(answer_key)} questions.")
        notify("A6", "done", questions=questions, answer_key=answer_key)

        _log("\n=== ORCHESTRATION COMPLETE ===\n")
        return {"questions": questions, "answer_key": answer_key}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_clo_mapping(self, questions: List[Dict], max_iterations: int,
                         on_step=None) -> List[Dict]:
        for iteration in range(max_iterations):
            mapped, coverage_ok = self._cm.map_and_check(questions)
            if on_step:
                try:
                    on_step("A4", "iteration",
                            {"iteration": iteration + 1,
                             "coverage_ok": coverage_ok,
                             "questions": mapped})
                except Exception:
                    pass
            if coverage_ok:
                _log(f"     Full CLO coverage achieved (iteration {iteration + 1}).")
                return mapped

            missing = self._get_missing_clos(mapped)
            _log(f"     Missing CLOs: {missing}. Generating gap-fill questions...")

            new_qs = self._generate_for_missing_clos(missing)
            classified = self._bc.classify(new_qs)
            balanced = self._db.balance(classified)
            questions = mapped + balanced

        _log("     Max iterations reached. Using best-effort coverage.")
        return mapped

    def _get_missing_clos(self, mapped_questions: List[Dict]) -> List[str]:
        covered = {q["clo"] for q in mapped_questions if q.get("clo") != "UNMAPPED"}
        all_clos = {doc.metadata["code"] for doc in self._cm.clos}
        return list(all_clos - covered)

    def _generate_for_missing_clos(self, missing_clos: List[str]) -> List[Dict]:
        new_questions: List[Dict] = []
        for clo in missing_clos:
            clo_desc = next(
                (doc.page_content for doc in self._cm.clos
                 if doc.metadata["code"] == clo),
                clo,
            )
            _log(f"     Targeting {clo}: {clo_desc}")
            qs = self._qg.generate_questions(clo_desc, num_questions=1)
            new_questions.extend(qs)
        return new_questions
