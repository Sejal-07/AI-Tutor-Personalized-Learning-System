from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np


class PerformanceAnalyzer:
    def __init__(self):
        self.concept_mastery_threshold = 70
        self.trend_models = {}
        self.mastery_classifiers = {}

    # --------------------------------------------------
    # 1️⃣ CONCEPT MASTERY LEVEL ANALYSIS (Rule-Based)
    # --------------------------------------------------
    def analyze_concept_mastery(self, student_features):
        """
        Assign mastery levels based on mastery_score
        """

        if student_features is None or student_features.empty:
            print("⚠️ No student features available for mastery analysis")
            return student_features

        mastery_levels = []

        for _, row in student_features.iterrows():
            score = row.get("mastery_score", 0)

            if score >= 85:
                level = "Advanced"
            elif score >= 70:
                level = "Intermediate"
            elif score >= 50:
                level = "Beginner"
            else:
                level = "Struggling"

            mastery_levels.append(level)

        student_features["mastery_level"] = mastery_levels
        return student_features

    # --------------------------------------------------
    # 2️⃣ PERFORMANCE TREND PREDICTION (Linear Regression)
    # --------------------------------------------------
    def predict_performance_trend(self, student_id, concept_id, performance_history):
        """
        Predict future performance using Linear Regression
        """

        if performance_history is None or len(performance_history) < 3:
            return None

        X = np.arange(len(performance_history)).reshape(-1, 1)
        y = np.array(performance_history)

        model = LinearRegression()
        model.fit(X, y)

        next_point = len(performance_history)
        predicted_score = model.predict([[next_point]])[0]

        return max(0, min(100, predicted_score))

    # --------------------------------------------------
    # 3️⃣ CONCEPT MASTERY CLASSIFICATION (ML-Based)
    # --------------------------------------------------
    def classify_concept_mastery(self, features):
        """
        Classify whether a concept is mastered using Logistic Regression
        """

        required_columns = [
            "accuracy",
            "avg_time_taken",
            "avg_attempts",
            "total_questions",
            "mastery_score",
        ]

        # ✅ 1. Validate dataframe
        if features is None or features.empty:
            print("⚠️ No data available for mastery classification")
            return features, 0.0

        # ✅ 2. Validate columns
        missing_cols = [col for col in required_columns if col not in features.columns]
        if missing_cols:
            print(f"⚠️ Missing required columns: {missing_cols}")
            return features, 0.0

        # ✅ 3. Prepare features & labels
        X = features[["accuracy", "avg_time_taken", "avg_attempts", "total_questions"]]
        y = (features["mastery_score"] >= self.concept_mastery_threshold).astype(int)

        # ✅ 4. Handle missing values
        X = X.fillna(X.mean())

        # ✅ 5. Cold-start / small data handling
        if len(X) < 5:
            print("⚠️ Insufficient samples – using rule-based fallback")
            features["predicted_mastery"] = y
            features["mastery_probability"] = y.astype(float)
            return features, 0.0

        # ✅ 6. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ✅ 7. Train classifier
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)

        # ✅ 8. Predict for all samples
        features["predicted_mastery"] = model.predict(X)
        features["mastery_probability"] = model.predict_proba(X)[:, 1]

        return features, model.score(X_test, y_test)

    # --------------------------------------------------
    # 4️⃣ WEAK CONCEPT IDENTIFICATION
    # --------------------------------------------------
    def get_weak_concepts(self, student_id, student_features):
        """
        Identify weakest concepts for a given student
        """

        if student_features is None or student_features.empty:
            return None

        student_data = student_features[student_features["student_id"] == student_id]

        if student_data.empty:
            return None

        weak_concepts = student_data[
            student_data["mastery_level"].isin(["Struggling", "Beginner"])
        ]

        weak_concepts = weak_concepts.sort_values("mastery_score")

        return weak_concepts[
            ["concept_id", "concept_name", "mastery_score", "mastery_level"]
        ]
