"""
Agent 4 — CLO/PLO Mapper (Semantic Dual Mapping)

Core Responsibilities:
✔ Maps Question → CLO (embedding similarity)
✔ Maps Question → PLO (embedding similarity)
✔ Detects uncovered CLOs
✔ No dependency on manual CLO→PLO mapping
"""

from typing import List, Dict, Tuple
import numpy as np
from rag_pipeline import get_all_clo_plo_docs


class CLOMapperAgent:
    def __init__(self, embeddings, similarity_threshold: float = 0.65):
        self.embeddings = embeddings
        self.threshold = similarity_threshold

        # Load all CLO/PLO docs
        self.outcome_docs = get_all_clo_plo_docs(embeddings)

        # Separate CLOs and PLOs
        self.clos = [
            d for d in self.outcome_docs if d.metadata.get("type") == "CLO"]
        self.plos = [
            d for d in self.outcome_docs if d.metadata.get("type") == "PLO"]

        if not self.clos:
            raise ValueError("No CLO documents found. Please ingest CLO/PLO data first.")
        if not self.plos:
            raise ValueError("No PLO documents found. Please ingest CLO/PLO data first.")

        # Precompute embeddings
        self.clo_texts = [doc.page_content for doc in self.clos]
        self.plo_texts = [doc.page_content for doc in self.plos]

        self.clo_embeddings = self.embeddings.embed_documents(self.clo_texts)
        self.plo_embeddings = self.embeddings.embed_documents(self.plo_texts)
        self.clo_to_plo = self._build_clo_plo_links()

    # ------------------------------------------------------------------
    # Main function
    # ------------------------------------------------------------------

    def map_and_check(self, questions: List[Dict]) -> Tuple[List[Dict], bool]:
        """
        Returns:
            mapped_questions, coverage_ok
        """

        mapped_questions = []
        covered_clos = set()

        for q in questions:
            # --- CLO Mapping ---
            clo_code, clo_score = self._match(
                q["question"], self.clos, self.clo_embeddings)

            # --- PLO Mapping ---
            # Prefer PLO linked to mapped CLO (more stable for OBE reporting),
            # fallback to direct question->PLO semantic match.
            direct_plo_code, direct_plo_score = self._match(
                q["question"], self.plos, self.plo_embeddings)

            linked_plo_code = self.clo_to_plo.get(clo_code)
            if linked_plo_code:
                plo_code = linked_plo_code
                plo_score = direct_plo_score
            else:
                plo_code = direct_plo_code
                plo_score = direct_plo_score

            # Apply threshold
            q["clo"] = clo_code if clo_score >= self.threshold else "UNMAPPED"
            q["plo"] = plo_code if plo_score >= self.threshold else "UNMAPPED"

            q["clo_similarity"] = round(clo_score, 3)
            q["plo_similarity"] = round(plo_score, 3)

            if q["clo"] != "UNMAPPED":
                covered_clos.add(q["clo"])

            mapped_questions.append(q)

        # Coverage check (ONLY for CLOs)
        all_clo_codes = {doc.metadata["code"] for doc in self.clos}
        uncovered = all_clo_codes - covered_clos

        print("\n[INFO] CLO Coverage Report:")
        print("Covered:", covered_clos)
        print("Uncovered:", uncovered)

        coverage_ok = len(uncovered) == 0
        return mapped_questions, coverage_ok

    # ------------------------------------------------------------------
    # Similarity computation
    # ------------------------------------------------------------------

    def _match(self, text: str, docs, embeddings):
        q_embedding = self.embeddings.embed_query(text)

        scores = [self._hybrid_score(text, q_embedding, doc.page_content, emb)
                  for doc, emb in zip(docs, embeddings)]

        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]
        best_code = docs[best_idx].metadata["code"]

        return best_code, best_score

    # ------------------------------------------------------------------
    # Cosine similarity
    # ------------------------------------------------------------------

    def _cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _token_overlap(self, a: str, b: str) -> float:
        a_tokens = {t.lower() for t in a.split() if len(t.strip()) > 2}
        b_tokens = {t.lower() for t in b.split() if len(t.strip()) > 2}
        if not a_tokens or not b_tokens:
            return 0.0
        inter = len(a_tokens & b_tokens)
        union = len(a_tokens | b_tokens)
        return inter / union

    def _hybrid_score(self, q_text: str, q_embedding, target_text: str, target_embedding) -> float:
        sem = self._cosine_similarity(q_embedding, target_embedding)
        lex = self._token_overlap(q_text, target_text)
        # Weighted blend: semantic anchors meaning, lexical overlap improves precision.
        return (0.85 * sem) + (0.15 * lex)

    def _build_clo_plo_links(self) -> Dict[str, str]:
        links = {}
        for clo_doc, clo_emb in zip(self.clos, self.clo_embeddings):
            scores = [self._cosine_similarity(clo_emb, plo_emb)
                      for plo_emb in self.plo_embeddings]
            best_idx = int(np.argmax(scores))
            clo_code = clo_doc.metadata["code"]
            plo_code = self.plos[best_idx].metadata["code"]
            links[clo_code] = plo_code
        return links
