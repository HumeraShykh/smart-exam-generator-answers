"""
Agent 3 — Difficulty Balancer

Responsibilities:
✔ Assign difficulty based on Bloom's level
✔ Enforce target distribution (30/40/30)
✔ Handle imbalance (duplicate or trim)
✔ Keep pipeline stable (no crashes)
"""

from typing import List, Dict
import random


class DifficultyBalancerAgent:
    def __init__(self):
        # Mapping Bloom → Difficulty
        self.bloom_to_difficulty = {
            "Remember": "Easy",
            "Understand": "Easy",
            "Apply": "Medium",
            "Analyze": "Medium",
            "Evaluate": "Hard",
            "Create": "Hard",
        }

        # Target distribution
        self.target_ratio = {
            "Easy": 0.3,
            "Medium": 0.4,
            "Hard": 0.3,
        }

    # ------------------------------------------------------------------
    # Main function
    # ------------------------------------------------------------------

    def balance(self, questions: List[Dict]) -> List[Dict]:
        """
        Input: classified questions
        Output: balanced questions with difficulty labels
        """

        # Step 1: assign difficulty
        for q in questions:
            bloom = q.get("blooms_level", "Understand")
            q["difficulty"] = self.bloom_to_difficulty.get(bloom, "Medium")

        # Step 2: group by difficulty
        groups = {
            "Easy": [],
            "Medium": [],
            "Hard": []
        }

        for q in questions:
            groups[q["difficulty"]].append(q)

        total = len(questions)

        # Step 3: calculate target counts
        target_counts = {
            level: max(1, int(self.target_ratio[level] * total))
            for level in self.target_ratio
        }

        print("\n[INFO] Target distribution:", target_counts)

        # Step 4: rebalance
        balanced = []

        for level in ["Easy", "Medium", "Hard"]:
            current = groups[level]
            target = target_counts[level]

            if len(current) >= target:
                # Trim extra
                selected = current[:target]
            else:
                # Not enough → duplicate existing
                selected = current.copy()

                while len(selected) < target and current:
                    selected.append(random.choice(current))

            balanced.extend(selected)

        print(f"[OK] Balanced total questions: {len(balanced)}")

        return balanced
