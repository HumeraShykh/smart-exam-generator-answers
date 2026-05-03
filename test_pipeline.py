"""
TEST PIPELINE (Agent 1 + Agent 2)

This script:
1. Builds RAG pipeline
2. Generates questions (Agent 1)
3. Classifies them (Agent 2)
4. Prints structured output

Make sure:
- LM Studio server is running
- Model is loaded (mistral-7b-instruct-v0.3)
- Embedding model is loaded
"""

from rag_pipeline import build_rag_pipeline
from agents.question_generator import QuestionGeneratorAgent
from agents.bloom_classifier import BloomClassifierAgent
from agents.difficulty_balancer import DifficultyBalancerAgent
from agents.clo_mapper import CLOMapperAgent


# ------------------------------------------------------------------
# STEP 1 — Build RAG pipeline
# ------------------------------------------------------------------

pipeline = build_rag_pipeline(
    syllabus_path="data/Lectures",
    clo_plo_path="data/clo_plo.yaml",
    past_papers_path="data/Assessment",
)

llm = pipeline["llm"]
embeddings = pipeline["embeddings"]


# ------------------------------------------------------------------
# STEP 2 — Initialize Agents
# ------------------------------------------------------------------

qg_agent = QuestionGeneratorAgent(llm=llm, embeddings=embeddings)
bc_agent = BloomClassifierAgent(llm=llm)


# ------------------------------------------------------------------
# STEP 3 — Generate Questions (Agent 1)
# ------------------------------------------------------------------

print("\n==============================")
print("AGENT 1: QUESTION GENERATION")
print("==============================\n")

topic = "Transformer architectures in NLP"

questions = qg_agent.generate_questions(topic, num_questions=6)

for i, q in enumerate(questions, 1):
    print(f"{i}. {q}")


# ------------------------------------------------------------------
# STEP 4 — Bloom Classification (Agent 2)
# ------------------------------------------------------------------

print("\n==============================")
print("AGENT 2: BLOOM CLASSIFICATION")
print("==============================\n")

classified_questions = bc_agent.classify(questions)

for i, q in enumerate(classified_questions, 1):
    print(f"{i}. {q}")


# ------------------------------------------------------------------
# OPTIONAL: Pretty print (cleaner view)
# ------------------------------------------------------------------

print("\n==============================")
print("FINAL STRUCTURED OUTPUT")
print("==============================\n")

for i, q in enumerate(classified_questions, 1):
    print(f"Q{i}: {q['question']}")
    print(f"   Type: {q['type']}")
    print(f"   Bloom Level: {q['blooms_level']}")
    print("-" * 50)

# ------------------------------------------------------------------
# STEP 5 — Difficulty Balancer (Agent 3)
# ------------------------------------------------------------------

db_agent = DifficultyBalancerAgent()

print("\n==============================")
print("AGENT 3: DIFFICULTY BALANCING")
print("==============================\n")

balanced_questions = db_agent.balance(classified_questions)

for i, q in enumerate(balanced_questions, 1):
    print(f"{i}. {q}")


# ------------------------------------------------------------------
# OPTIONAL: Pretty print (cleaner view)
# ------------------------------------------------------------------

print("\n==============================")
print("FINAL STRUCTURED OUTPUT")
print("==============================\n")

for i, q in enumerate(balanced_questions, 1):
    print(f"Q{i}: {q['question']}")
    print(f"   Type: {q['type']}")
    print(f"   Bloom Level: {q['blooms_level']}")
    print("-" * 50)


# ------------------------------------------------------------------
# STEP 5 — Difficulty Balancer (Agent 3)
# ------------------------------------------------------------------

cm_agent = CLOMapperAgent(embeddings=embeddings)

print("\n==============================")
print("AGENT 4: CLO MAPPING")
print("==============================\n")

mapped_questions, coverage_ok = cm_agent.map_and_check(balanced_questions)

for i, q in enumerate(mapped_questions, 1):
    print(f"{i}. {q}")

print("\nCoverage OK:", coverage_ok)
