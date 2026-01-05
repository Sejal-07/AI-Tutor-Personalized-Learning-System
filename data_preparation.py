# data_preparation.py
import pandas as pd
import numpy as np

# --------------------------------------------------
# Load datasets
# --------------------------------------------------
learning_resources = pd.read_csv("data/learning_resources_full.csv")
questions = pd.read_csv("data/question_table.csv")
concepts = pd.read_csv("data/concept_table.csv")
students = pd.read_csv("data/student_table.csv")
performance = pd.read_csv("data/student_performance.csv")


# --------------------------------------------------
# Phase 1: Data Preparation & Feature Engineering
# --------------------------------------------------
def prepare_data():
    """
    Prepare student–concept level performance features
    """

    # 1️⃣ Merge performance with questions
    performance_with_concepts = performance.merge(
        questions[["question_id", "concept_id", "difficulty"]],
        on="question_id",
        how="left",
    )

    # 2️⃣ Merge with concept information
    performance_full = performance_with_concepts.merge(
        concepts[["concept_id", "concept_name", "subject", "level"]],
        on="concept_id",
        how="left",
    )

    # 3️⃣ Aggregate performance per student per concept
    student_concept_performance = (
        performance_full.groupby(["student_id", "concept_id"])
        .agg(
            accuracy=("correct", "mean"),
            total_questions=("correct", "count"),
            correct_count=("correct", "sum"),
            avg_time_taken=("time_taken", "mean"),
            time_std=("time_taken", "std"),
            avg_attempts=("attempts", "mean"),
        )
        .reset_index()
    )

    # 4️⃣ Merge with student profile data
    student_features = student_concept_performance.merge(
        students[["student_id", "learning_style", "avg_accuracy", "avg_response_time"]],
        on="student_id",
        how="left",
    )

    # 5️⃣ Compute mastery score
    def calculate_mastery_score(row):
        base_score = row["accuracy"] * 100

        # Penalize slow responses
        time_penalty = 0
        if row["avg_time_taken"] > 60:
            time_penalty = min(10, (row["avg_time_taken"] - 60) / 10)

        # Penalize multiple attempts
        attempt_penalty = 0
        if row["avg_attempts"] > 1.5:
            attempt_penalty = min(5, (row["avg_attempts"] - 1) * 2)

        return max(0, base_score - time_penalty - attempt_penalty)

    student_features["mastery_score"] = student_features.apply(
        calculate_mastery_score, axis=1
    )

    # 6️⃣ Flag weak concepts
    student_features["is_weak_concept"] = student_features["mastery_score"] < 70

    return student_features, performance_full, learning_resources, concepts, students


# --------------------------------------------------
# Student-level Feature Builder (FOR CLUSTERING)
# --------------------------------------------------
def build_student_level_features(student_concept_features):
    """
    Convert student–concept level data into student-level features
    """

    if student_concept_features is None or student_concept_features.empty:
        print("⚠️ No student–concept data available")
        return None

    student_level_features = (
        student_concept_features.groupby("student_id")
        .agg(
            accuracy=("accuracy", "mean"),
            avg_time_taken=("avg_time_taken", "mean"),
            avg_attempts=("avg_attempts", "mean"),
            total_questions=("total_questions", "sum"),
            mastery_score=("mastery_score", "mean"),
            weak_concept_ratio=("is_weak_concept", "mean"),
        )
        .reset_index()
    )

    return student_level_features
