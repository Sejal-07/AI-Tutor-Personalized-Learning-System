# main.py
import warnings

warnings.filterwarnings("ignore")

from data_preparation import prepare_data, build_student_level_features
from performance_analysis import PerformanceAnalyzer
from student_clustering import StudentClustering
from similarity_engine import SimilarityEngine
from recommendation_engine import RecommendationEngine


def main():
    print("=" * 60)
    print("AI Tutor - Personalized Learning Recommendation System")
    print("=" * 60)

    # --------------------------------------------------
    # Phase 1: Data Preparation & Feature Engineering
    # --------------------------------------------------
    print("\n📊 Phase 1: Data Preparation & Feature Engineering")
    print("-" * 50)

    (
        student_concept_features,
        performance_data,
        resources,
        concepts,
        students,
    ) = prepare_data()

    print(f"✓ Students loaded: {students.shape[0]}")
    print(f"✓ Concepts loaded: {concepts.shape[0]}")
    print(f"✓ Resources loaded: {resources.shape[0]}")
    print(f"✓ Student–Concept rows: {student_concept_features.shape[0]}")

    # Add concept names (needed later)
    student_concept_features = student_concept_features.merge(
        concepts[["concept_id", "concept_name"]],
        on="concept_id",
        how="left",
    )

    # --------------------------------------------------
    # Phase 2: Performance Analysis (Concept-Level)
    # --------------------------------------------------
    print("\n📈 Phase 2: Performance Analysis")
    print("-" * 50)

    analyzer = PerformanceAnalyzer()

    student_concept_features = analyzer.analyze_concept_mastery(
        student_concept_features
    )

    student_concept_features, classifier_score = analyzer.classify_concept_mastery(
        student_concept_features
    )

    print("✓ Concept mastery analyzed")
    print(f"✓ Classifier accuracy: {classifier_score:.2%}")

    # --------------------------------------------------
    # Phase 3: Student-Level Aggregation & Clustering
    # --------------------------------------------------
    print("\n👥 Phase 3: Student Clustering")
    print("-" * 50)

    # 🔑 Build student-level features
    student_level_features = build_student_level_features(student_concept_features)

    print(f"✓ Student-level feature rows: {student_level_features.shape[0]}")

    clustering = StudentClustering(n_clusters=4)

    # Extract clustering features
    clustering_features = student_level_features[
        [
            "accuracy",
            "avg_time_taken",
            "avg_attempts",
            "total_questions",
            "mastery_score",
        ]
    ]

    # Apply clustering
    cluster_labels = clustering.apply_clustering(clustering_features)

    # Analyze clusters (IMPORTANT — adds cluster_name)
    student_level_features, cluster_stats = clustering.analyze_clusters(
        student_level_features, cluster_labels
    )

    print(f"✓ Students clustered into {clustering.n_clusters} groups")
    print("\nCluster Distribution:")
    print(student_level_features["cluster_name"].value_counts())

    # Visualize AFTER analysis
    clustering.visualize_clusters(student_level_features, clustering_features)

    # --------------------------------------------------
    # Phase 4: Similarity Engine
    # --------------------------------------------------
    print("\n🔍 Phase 4: Similarity Engine")
    print("-" * 50)

    similarity_engine = SimilarityEngine(students)

    all_concepts = concepts["concept_id"].unique()

    student_vectors = similarity_engine.create_student_vectors(
        student_concept_features, all_concepts
    )

    similarity_matrix = similarity_engine.compute_similarity()

    print(f"✓ Similarity matrix created for {len(student_vectors)} students")

    # --------------------------------------------------
    # Phase 5: Recommendation Engine
    # --------------------------------------------------
    print("\n🎯 Phase 5: Recommendation Engine")
    print("-" * 50)

    recommender = RecommendationEngine(
        resources=resources,
        concepts=concepts,
        students=students,
        student_features=student_concept_features,
        analyzer=analyzer,
        similarity_engine=similarity_engine,
    )

    sample_student = students["student_id"].iloc[0]
    print(f"\nGenerating personalized plan for student: {sample_student}")

    personalized_plan = recommender.generate_personalized_plan(sample_student)

    print("✓ Personalized learning plan generated")
    print(f"✓ Weak concepts identified: {personalized_plan['weak_concepts_count']}")
    print(f"✓ Recommendations created: {len(personalized_plan['recommendations'])}")

    # --------------------------------------------------
    # Phase 6: System Ready
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("✅ SYSTEM INITIALIZATION COMPLETE")
    print("=" * 60)

    return {
        "student_concept_features": student_concept_features,
        "student_level_features": student_level_features,
        "students": students,
        "concepts": concepts,
        "resources": resources,
        "analyzer": analyzer,
        "similarity_engine": similarity_engine,
        "recommender": recommender,
        "sample_plan": personalized_plan,
        "cluster_stats": cluster_stats,
    }


if __name__ == "__main__":
    system_components = main()
