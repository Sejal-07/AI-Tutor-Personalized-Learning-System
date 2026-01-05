# similarity_engine.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityEngine:
    def __init__(self, students_df: pd.DataFrame):
        self.students = students_df
        self.student_vectors = None
        self.similarity_matrix = None

    # --------------------------------------------------
    # Create student × concept mastery vectors
    # --------------------------------------------------
    def create_student_vectors(self, student_concept_df, all_concepts):
        """
        Creates a student × concept mastery matrix
        """

        student_vectors = student_concept_df.pivot_table(
            index="student_id",
            columns="concept_id",
            values="mastery_score",
            aggfunc="mean",
        )

        # Ensure all concepts are present
        for concept_id in all_concepts:
            if concept_id not in student_vectors.columns:
                student_vectors[concept_id] = 0.0

        student_vectors = student_vectors.fillna(0.0).astype(float)

        self.student_vectors = student_vectors

        print(f"✓ Student vectors created: {student_vectors.shape}")
        return student_vectors

    # --------------------------------------------------
    # Compute cosine similarity matrix
    # --------------------------------------------------
    def compute_similarity(self):
        """
        Computes cosine similarity between students
        """

        if self.student_vectors is None or self.student_vectors.empty:
            raise ValueError(
                "❌ Student vectors not initialized. Call create_student_vectors() first."
            )

        self.similarity_matrix = cosine_similarity(self.student_vectors.values)

        print(f"✓ Similarity matrix created: {self.similarity_matrix.shape}")
        return self.similarity_matrix

    # --------------------------------------------------
    # Find similar students (ALWAYS returns tuples)
    # --------------------------------------------------
    def find_similar_students(self, student_id, top_k=5, top_n=None):
        """
        Returns: List of (peer_id, similarity_score)
        """

        if self.similarity_matrix is None:
            self.compute_similarity()

        if student_id not in self.student_vectors.index:
            raise ValueError(f"❌ Student ID {student_id} not found")

        # Alias handling
        if top_n is not None:
            top_k = top_n

        student_idx = self.student_vectors.index.get_loc(student_id)
        similarity_scores = self.similarity_matrix[student_idx]

        results = []
        for idx, score in enumerate(similarity_scores):
            if idx != student_idx:
                peer_id = self.student_vectors.index[idx]
                results.append((peer_id, float(score)))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[: int(top_k)]

    # --------------------------------------------------
    # Peer pattern analysis (used by recommender)
    # --------------------------------------------------
    def analyze_peer_patterns(self, student_id, top_k=5, mastery_threshold=70):
        """
        Returns:
        {
            peer_id: {
                "similarity": float,
                "strong_concepts": [
                    {"concept_id": ..., "mastery_score": ...}
                ]
            }
        }
        """

        similar_students = self.find_similar_students(
            student_id=student_id, top_k=top_k
        )

        if not similar_students:
            return {}

        peer_patterns = {}

        for peer_id, similarity in similar_students:
            peer_mastery = self.student_vectors.loc[peer_id]

            strong_concepts = [
                {
                    "concept_id": concept_id,
                    "mastery_score": float(score),
                }
                for concept_id, score in peer_mastery.items()
                if score >= mastery_threshold
            ]

            peer_patterns[peer_id] = {
                "similarity": similarity,
                "strong_concepts": strong_concepts[:5],
            }

        return peer_patterns
