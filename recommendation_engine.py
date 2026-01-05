# recommendation_engine.py
import pandas as pd
from similarity_engine import SimilarityEngine


class RecommendationEngine:
    def __init__(
        self,
        resources,
        concepts,
        student_features,
        analyzer,
        similarity_engine,
        students,
    ):
        self.resources = resources
        self.concepts = concepts
        self.student_features = student_features
        self.analyzer = analyzer
        self.similarity_engine = similarity_engine
        self.students = students

    # --------------------------------------------------
    # Content-based recommendations (FIXED)
    # --------------------------------------------------
    def get_content_based_recommendations(self, student_id, weak_concepts, top_n=3):
        recommendations = []

        for _, weak_concept in weak_concepts.iterrows():
            concept_id = weak_concept.get("concept_id")
            concept_name = weak_concept.get("concept_name")
            mastery_level = weak_concept.get("mastery_level", "Beginner")

            # 1️⃣ Primary: concept_id match
            concept_resources = self.resources[
                self.resources["concept_id"] == concept_id
            ]

            # 2️⃣ SAFE fallback: plain string match (NO REGEX)
            if concept_resources.empty:
                if isinstance(concept_name, str) and concept_name.strip():
                    concept_resources = self.resources[
                        self.resources["concept_name"]
                        .astype(str)
                        .str.contains(
                            concept_name,
                            case=False,
                            na=False,
                            regex=False,
                        )
                    ]

            if concept_resources.empty:
                continue

            # 3️⃣ Difficulty filtering
            if mastery_level == "Struggling":
                suitable = concept_resources[
                    concept_resources["difficulty"] == "Beginner"
                ]
            elif mastery_level == "Beginner":
                suitable = concept_resources[
                    concept_resources["difficulty"].isin(["Beginner", "Intermediate"])
                ]
            else:
                suitable = concept_resources

            suitable = suitable.sort_values(
                ["rating", "view_count"],
                ascending=[False, False],
            ).head(top_n)

            for _, resource in suitable.iterrows():
                recommendations.append(
                    {
                        "concept_id": concept_id,
                        "concept_name": concept_name,
                        "resource_id": resource["resource_id"],
                        "resource_name": resource["resource_name"],
                        "resource_type": resource["resource_type"],
                        "difficulty": resource["difficulty"],
                        "duration_minutes": resource["duration_minutes"],
                        "rating": resource["rating"],
                        "url": resource["url"],
                        "reason": f"Low mastery in {concept_name} ({weak_concept['mastery_score']:.1f}%)",
                    }
                )

        return recommendations[:10]

    # --------------------------------------------------
    # Behavior-based recommendations
    # --------------------------------------------------
    def get_behavior_based_recommendations(
        self, student_id, similar_students, peer_patterns
    ):
        recommendations = []

        for peer_id, peer_info in peer_patterns.items():
            similarity = peer_info.get("similarity", 0)

            for strong_concept in peer_info.get("strong_concepts", []):
                concept_id = strong_concept["concept_id"]

                mastery = self.student_features[
                    (self.student_features["student_id"] == student_id)
                    & (self.student_features["concept_id"] == concept_id)
                ]

                if not mastery.empty and mastery["mastery_score"].iloc[0] >= 80:
                    continue

                peer_resources = self.resources[
                    self.resources["concept_id"] == concept_id
                ].sort_values("rating", ascending=False)

                if peer_resources.empty:
                    continue

                best = peer_resources.iloc[0]

                recommendations.append(
                    {
                        "concept_id": concept_id,
                        "concept_name": strong_concept.get("concept_name", concept_id),
                        "resource_id": best["resource_id"],
                        "resource_name": best["resource_name"],
                        "resource_type": best["resource_type"],
                        "difficulty": best["difficulty"],
                        "duration_minutes": best["duration_minutes"],
                        "rating": best["rating"],
                        "url": best["url"],
                        "reason": f"Used by similar student {peer_id} (similarity {similarity:.2f})",
                    }
                )

        return recommendations

    # --------------------------------------------------
    # Learning path
    # --------------------------------------------------
    def create_learning_path(self, student_id, weak_concepts):
        path = []
        visited = set()

        enriched = []
        for _, row in weak_concepts.iterrows():
            info = self.concepts[self.concepts["concept_id"] == row["concept_id"]]
            if info.empty:
                continue

            enriched.append(
                {
                    **row.to_dict(),
                    "level": info["level"].iloc[0],
                    "prerequisite_id": info["prerequisite_id"].iloc[0],
                }
            )

        enriched.sort(key=lambda x: (x["level"], x["mastery_score"]))

        for concept in enriched:
            cid = concept["concept_id"]
            if cid in visited:
                continue

            path.append(
                {
                    "type": "main",
                    "concept_id": cid,
                    "concept_name": concept["concept_name"],
                    "current_mastery": concept["mastery_score"],
                    "target_mastery": 80,
                }
            )
            visited.add(cid)

        return path

    # --------------------------------------------------
    # Master pipeline
    # --------------------------------------------------
    def generate_personalized_plan(self, student_id):
        weak_concepts = self.analyzer.get_weak_concepts(
            student_id, self.student_features
        )

        similar_students = self.similarity_engine.find_similar_students(student_id)
        peer_patterns = self.similarity_engine.analyze_peer_patterns(student_id)

        content_recs = self.get_content_based_recommendations(student_id, weak_concepts)

        behavior_recs = self.get_behavior_based_recommendations(
            student_id, similar_students, peer_patterns
        )

        recommendations = {
            (r["concept_id"], r["resource_id"]): r
            for r in (content_recs + behavior_recs)
        }

        learning_style = self.students[self.students["student_id"] == student_id][
            "learning_style"
        ].iloc[0]

        schedule = self.create_study_schedule(
            list(recommendations.values()), learning_style
        )

        return {
            "student_id": student_id,
            "learning_style": learning_style,
            "weak_concepts_count": len(weak_concepts),
            "learning_path": self.create_learning_path(student_id, weak_concepts),
            "recommendations": list(recommendations.values())[:5],
            "study_schedule": schedule,
        }

    # --------------------------------------------------
    # Study schedule
    # --------------------------------------------------
    def create_study_schedule(self, recommendations, learning_style):
        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        schedule = {d: [] for d in days}

        for i, rec in enumerate(recommendations[:7]):
            day = days[i % 7]
            schedule[day].append(
                {
                    "concept": rec["concept_name"],
                    "activity": rec["resource_type"],
                    "duration": f"{rec['duration_minutes']} minutes",
                    "resource": rec["resource_name"],
                }
            )

        return schedule
