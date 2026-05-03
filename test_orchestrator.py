from rag_pipeline import build_rag_pipeline
from orchestrator import ExamOrchestrator


pipeline = build_rag_pipeline(
    syllabus_path="data/Lectures",
    clo_plo_path="data/clo_plo.yaml",
    past_papers_path="data/Assessment",
)

orchestrator = ExamOrchestrator(pipeline)

final_questions = orchestrator.run(
    topic="Transformer architectures in NLP",
    max_iterations=3
)

print("\n==============================")
print("FINAL OUTPUT")
print("==============================\n")

for i, q in enumerate(final_questions, 1):
    print(f"Q{i}: {q}")

# ------------------------------------------------------------------
# OPTIONAL: Pretty print (cleaner view)
# ------------------------------------------------------------------

print("\n==============================")
print("FINAL STRUCTURED OUTPUT")
print("==============================\n")

for i, q in enumerate(final_questions, 1):
    print(f"Q{i}: {q['question']}")
    print(f"   Type: {q['type']}")
    print(f"   Bloom Level: {q['blooms_level']}")
    print("-" * 50)
