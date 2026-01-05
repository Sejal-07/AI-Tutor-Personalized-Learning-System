# app.py
from flask import Flask, request, jsonify
from main import main

app = Flask(__name__)

# 🔹 Load system ONCE at startup
system = main()

student_features = system["student_features"]
student_agg_data = system["student_agg"]
resources = system["resources"]
concepts = system["concepts"]
recommender = system["recommender"]


@app.route("/api/student/<student_id>/recommendations", methods=["GET"])
def get_recommendations(student_id):
    try:
        plan = recommender.generate_personalized_plan(student_id)

        plan["explanations"] = {
            "weak_concepts": "Identified weak concepts based on performance history",
            "learning_path": "Learning path follows prerequisite structure",
            "recommendations": "Resources selected based on difficulty & learning style",
            "schedule": f"Optimized for {plan.get('learning_style', 'default')} learning style",
        }

        return jsonify({"success": True, "student_id": student_id, "plan": plan})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/student/<student_id>/progress", methods=["GET"])
def get_progress(student_id):
    try:
        student_data = student_features[student_features["student_id"] == student_id]

        total_concepts = len(student_data)
        mastered_concepts = len(student_data[student_data["mastery_score"] >= 70])
        weak_concepts = len(student_data[student_data["mastery_score"] < 70])
        avg_mastery = (
            float(student_data["mastery_score"].mean()) if total_concepts > 0 else 0
        )

        if student_id in student_agg_data["student_id"].values:
            cluster_label = student_agg_data[
                student_agg_data["student_id"] == student_id
            ]["cluster_label"].iloc[0]
        else:
            cluster_label = "Not clustered"

        return jsonify(
            {
                "success": True,
                "student_id": student_id,
                "progress": {
                    "total_concepts_studied": total_concepts,
                    "mastered_concepts": mastered_concepts,
                    "weak_concepts": weak_concepts,
                    "average_mastery": avg_mastery,
                    "mastery_percentage": (
                        (mastered_concepts / total_concepts * 100)
                        if total_concepts > 0
                        else 0
                    ),
                },
                "cluster_info": {"cluster_label": cluster_label},
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/resources/search", methods=["GET"])
def search_resources():
    try:
        concept_id = request.args.get("concept_id")
        resource_type = request.args.get("resource_type")
        difficulty = request.args.get("difficulty")

        filtered_resources = resources.copy()

        if concept_id:
            filtered_resources = filtered_resources[
                filtered_resources["concept_id"] == concept_id
            ]
        if resource_type:
            filtered_resources = filtered_resources[
                filtered_resources["resource_type"] == resource_type
            ]
        if difficulty:
            filtered_resources = filtered_resources[
                filtered_resources["difficulty"] == difficulty
            ]

        filtered_resources = filtered_resources.sort_values(
            ["rating", "view_count"], ascending=[False, False]
        ).head(20)

        return jsonify(
            {
                "success": True,
                "count": len(filtered_resources),
                "resources": filtered_resources.to_dict("records"),
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    print("🚀 AI Tutor Recommendation System API Running")
    app.run(debug=True, port=5000)
