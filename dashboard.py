# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------
# Load your modules and data
# -------------------------------
from data_preparation import prepare_data
from performance_analysis import PerformanceAnalyzer
from recommendation_engine import RecommendationEngine
from similarity_engine import SimilarityEngine

# -------------------------------
# Load all data
# -------------------------------
student_features, performance_data, resources, concepts, students = prepare_data()

# Merge concept names into student_features for charts and recommendations
student_features = student_features.merge(
    concepts[["concept_id", "concept_name"]], on="concept_id", how="left"
)


# -------------------------------
# Add mastery_level column
# -------------------------------
def assign_mastery_level(score):
    if score >= 85:
        return "High"
    elif score >= 70:
        return "Medium"
    else:
        return "Low"


student_features["mastery_level"] = student_features["mastery_score"].apply(
    assign_mastery_level
)

# -------------------------------
# Initialize analyzer, similarity_engine, and recommender
# -------------------------------
analyzer = PerformanceAnalyzer()
similarity_engine = SimilarityEngine(students)

# Prepare similarity matrix
all_concepts = concepts["concept_id"].unique()
student_vectors = similarity_engine.create_student_vectors(
    student_features, all_concepts
)
similarity_matrix = similarity_engine.compute_similarity()

recommender = RecommendationEngine(
    resources=resources,
    concepts=concepts,
    students=students,
    student_features=student_features,
    analyzer=analyzer,
    similarity_engine=similarity_engine,
)

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="AI Tutor - Personalized Learning", page_icon="📚", layout="wide"
)

# -------------------------------
# Initialize session state
# -------------------------------
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "student_id" not in st.session_state:
    st.session_state.student_id = None

# -------------------------------
# Sidebar - Student Selection
# -------------------------------
st.sidebar.title("AI Tutor Dashboard")
st.sidebar.markdown("---")

student_options = students["student_id"].unique()
selected_student = st.sidebar.selectbox(
    "Select Student ID:", options=student_options, index=0
)
st.session_state.student_id = selected_student

# Get student info
student_info = students[students["student_id"] == selected_student].iloc[0]

# -------------------------------
# Main Content
# -------------------------------
st.title(f"📚 Personalized Learning Recommendations")
st.markdown(
    f"**Student:** {selected_student} | **Learning Style:** {student_info['learning_style']}"
)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "🎯 Recommendations", "📈 Progress", "👥 Similar Students"]
)

# -------------------------------
# Tab 1: Overview
# -------------------------------
with tab1:
    col1, col2, col3 = st.columns(3)
    student_data = student_features[student_features["student_id"] == selected_student]
    total_concepts = len(student_data)
    mastered = len(student_data[student_data["mastery_score"] >= 70])

    with col1:
        st.metric(
            label="Concepts Mastered",
            value=f"{mastered}/{total_concepts}",
            delta=f"{mastered/total_concepts*100:.1f}%" if total_concepts > 0 else "0%",
        )
    with col2:
        avg_mastery = student_data["mastery_score"].mean() if total_concepts > 0 else 0
        st.metric(label="Average Mastery", value=f"{avg_mastery:.1f}%", delta="+2.5%")
    with col3:
        avg_time = student_data["avg_time_taken"].mean() if total_concepts > 0 else 0
        st.metric(label="Avg Response Time", value=f"{avg_time:.1f}s", delta="-5.2s")

    st.subheader("Weak Concepts Analysis")
    weak_concepts = student_data[student_data["is_weak_concept"]].sort_values(
        "mastery_score"
    )
    if len(weak_concepts) > 0:
        fig = px.bar(
            weak_concepts,
            x="concept_name",
            y="mastery_score",
            color="mastery_score",
            title="Weak Concepts by Mastery Score",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("🎉 No weak concepts identified! Keep up the great work!")

# -------------------------------
# Tab 2: Recommendations
# -------------------------------
with tab2:
    st.subheader("Personalized Recommendations")
    if st.button("Generate New Recommendations", type="primary"):
        with st.spinner("Analyzing performance and generating recommendations..."):
            plan = recommender.generate_personalized_plan(selected_student)
            st.session_state.recommendations = plan

    if st.session_state.recommendations:
        plan = st.session_state.recommendations

        # Learning Path
        st.markdown("### 🗺️ Learning Path")
        for i, step in enumerate(plan.get("learning_path", [])[:5], 1):
            concept_name = step.get("concept_name", "Unknown Concept")
            step_type = step.get("type", "main")
            reason = step.get("reason", "No reason provided")
            current_mastery = step.get("current_mastery", 0)
            target_mastery = step.get("target_mastery", 70)

            with st.expander(f"Step {i}: {concept_name} ({step_type})"):
                st.write(f"**Reason:** {reason}")
                if step_type == "main":
                    st.write(f"**Current Mastery:** {current_mastery:.1f}%")
                    st.write(f"**Target Mastery:** {target_mastery}%")

        # Resource Recommendations
        st.markdown("### 📚 Recommended Resources")
        for i, rec in enumerate(plan.get("recommendations", [])[:3], 1):
            col1, col2 = st.columns([3, 1])
            with col1:
                resource_name = rec.get("resource_name", "Unknown Resource")
                concept_name = rec.get("concept_name", "Unknown Concept")
                resource_type = rec.get("resource_type", "Unknown Type")
                difficulty = rec.get("difficulty", "Unknown")
                duration = rec.get("duration_minutes", 0)
                rating = rec.get("rating", 0)
                reason = rec.get("reason", "No reason provided")
                url = rec.get("url", "#")

                st.markdown(f"**{i}. {resource_name}**")
                st.markdown(f"*{concept_name} | {resource_type} | {difficulty}*")
                st.markdown(f"Duration: {duration} min ⭐ Rating: {rating}")
                st.markdown(f"**Why:** {reason}")
            with col2:
                st.markdown(f"[Open Resource]({url})")

# -------------------------------
# Tab 3: Progress
# -------------------------------
with tab3:
    st.subheader("Performance Progress")
    if len(student_data) > 0:
        fig = px.pie(
            student_data,
            names="mastery_level",
            title="Mastery Level Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Detailed Concept Performance")
        display_cols = [
            "concept_name",
            "mastery_score",
            "mastery_level",
            "accuracy",
            "avg_time_taken",
        ]
        st.dataframe(
            student_data[display_cols].sort_values("mastery_score"),
            use_container_width=True,
        )

# -------------------------------
# Tab 4: Similar Students
# -------------------------------
with tab4:
    st.subheader("Similar Students Analysis")
    similar_students = similarity_engine.find_similar_students(
        selected_student, top_n=3
    )

    if similar_students:
        st.markdown("### Top 3 Similar Students")
        for peer_id, similarity in similar_students:
            peer_info = students[students["student_id"] == peer_id].iloc[0]
            peer_performance = student_features[
                student_features["student_id"] == peer_id
            ]

            with st.expander(
                f"{peer_id} (Similarity: {similarity:.2f}) - {peer_info['learning_style']} learner"
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Avg Accuracy", f"{peer_info['avg_accuracy']:.1f}%")
                with col2:
                    st.metric("Response Time", f"{peer_info['avg_response_time']:.1f}s")

                peer_strong = peer_performance[peer_performance["mastery_score"] > 80]
                if len(peer_strong) > 0:
                    st.markdown("**Strong Concepts:**")
                    for _, row in peer_strong.iterrows():
                        st.write(
                            f"- {row['concept_name']}: {row['mastery_score']:.1f}%"
                        )
    else:
        st.info(
            "Similarity analysis not yet performed. Generate recommendations first."
        )

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("**AI Tutor Recommendation System** | *Personalized Learning Paths*")
