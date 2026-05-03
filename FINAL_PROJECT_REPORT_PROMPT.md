# Final Project Report Prompt (Use in ChatGPT)

Copy this full prompt into ChatGPT and ask it to generate a professional academic report.

---

You are an academic report writer. Write a **professional Final Project Evaluation Report** for my project.

## 1) Project Context

- **Project Title:** Multi-Agent Exam Paper Generator using RAG
- **Domain:** NLP + Educational Assessment (OBE-aligned exam generation)
- **Tech Stack:** Python, Streamlit, LangChain, ChromaDB, LM Studio (local LLM + embeddings)
- **Core Idea:** The system takes syllabus/lecture material, CLO/PLO outcomes, and past papers, then generates exam questions and answer keys through a 6-agent pipeline.

## 2) System Overview

The architecture is:
1. Upload inputs:
   - Syllabus / lecture notes
   - CLO + PLO YAML file
   - Past papers + assignments
2. Build RAG pipeline:
   - Document loading (PDF/TXT/MD/PPTX)
   - Text cleaning
   - Chunking + embeddings
   - Storage in ChromaDB collections
3. Run orchestrator with 6 agents:
   - **Agent 1: Question Generator** (RAG-grounded question generation)
   - **Agent 2: Bloom Classifier** (Remember/Understand/Apply/Analyze/Evaluate/Create)
   - **Agent 3: Difficulty Balancer** (target distribution Easy/Medium/Hard)
   - **Agent 4: CLO/PLO Mapper** (semantic mapping + CLO gap-fill iterations)
   - **Agent 5: Quality Checker** (invalid/duplicate filtering)
   - **Agent 6: Answer Key Generator** (model answers + marking scheme + marks)
4. Final output:
   - Exam questions JSON
   - Answer key JSON
   - CLO/PLO alignment information

## 3) Write the Report Using These Required Sections

Please produce clear headings and detailed paragraphs for each of the following:

1. **Introduction and problem statement**
2. **Methodology and technical approach**
3. **Dataset preparation and usage**
4. **Working implementation**
5. **Results and evaluation**
6. **Conclusion and references**
7. **Quality of presentation and demonstration**

## 4) Content Requirements

When writing, include the following details:

- Explain why manual exam paper setting is time-consuming and hard to align with CLO/PLO and Bloom taxonomy.
- Explain how RAG reduces hallucination by grounding generation in uploaded course material and past papers.
- Mention that the app runs through a Streamlit UI and shows live multi-agent execution.
- Describe the role of each agent and how outputs flow sequentially.
- Explain CLO coverage gap-filling loop in Agent 4 (iterative regeneration until coverage improves).
- Mention robustness features:
  - JSON parsing recovery from imperfect LLM outputs
  - Duplicate filtering
  - Fallback behavior if parsing fails
- Explain local/offline-friendly deployment via LM Studio API.
- Discuss limitations honestly, for example:
  - Quality depends on uploaded data quality
  - Local model capability affects output quality
  - Difficulty balancing may duplicate when sample size is low
  - Semantic thresholds may require tuning

## 5) Results Section Instructions

In the **Results and evaluation** section:

- Provide a qualitative evaluation of generated questions (relevance, clarity, variety).
- Discuss Bloom-level coverage and difficulty distribution behavior.
- Discuss CLO/PLO mapping confidence (semantic similarity based).
- Mention practical usefulness for instructors.
- Add a short "observed challenges and improvements" subsection.

If numeric metrics are not available, present realistic qualitative evaluation and suggest future quantitative metrics (precision, coverage rate, duplicate rate, rubric quality score).

## 6) Tone and Formatting

- Formal academic tone (final year project report style)
- Clear and concise, no unnecessary repetition
- Use transition sentences between sections
- Add a short abstract at the start
- Add keywords after abstract
- Include a brief future work subsection in conclusion

## 7) References Guidance

At the end, include references in a consistent style (APA-like is fine), including:
- RAG / retrieval-augmented generation sources
- Bloom taxonomy source
- OBE/CLO-PLO assessment alignment literature
- LangChain / ChromaDB / Streamlit / LM Studio documentation references

If exact bibliographic metadata is uncertain, provide "web documentation" style references with links and access date placeholders.

## 8) Extra Output

After the full report, also provide:

1. **Presentation Outline (8-10 slides)** with slide titles and key bullet points.
2. **Demo Script (3-5 minutes)** for project presentation flow.

---

### Optional: My quick project summary for you to improve

This project is a multi-agent AI exam generator for NLP courses. It uses a RAG pipeline over syllabus, CLO/PLO, and past papers. A sequential orchestrator runs six agents to generate questions, classify Bloom levels, balance difficulty, map CLO/PLO with semantic similarity, filter quality, and generate answer keys with marking schemes. The system is implemented in Python with Streamlit front-end, LangChain orchestration utilities, ChromaDB vector store, and LM Studio local models.
