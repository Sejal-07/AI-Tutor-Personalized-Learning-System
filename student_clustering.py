# student_clustering.py

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class StudentClustering:
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=10
        )

    # -------------------------------------------------
    # 1. APPLY CLUSTERING
    # -------------------------------------------------
    def apply_clustering(self, features: pd.DataFrame) -> np.ndarray:
        """
        Scale features and apply K-Means clustering.
        """
        scaled_features = self.scaler.fit_transform(features)
        cluster_labels = self.kmeans.fit_predict(scaled_features)
        return cluster_labels

    # -------------------------------------------------
    # 2. ANALYZE & NAME CLUSTERS (FIXED LOGIC)
    # -------------------------------------------------
    def analyze_clusters(self, student_data: pd.DataFrame, cluster_labels: np.ndarray):
        """
        Analyze clusters and assign human-readable cluster names
        using RELATIVE (rank-based) logic.
        """

        student_data = student_data.copy()
        student_data["cluster_label"] = cluster_labels

        # Cluster-level statistics
        cluster_stats = (
            student_data.groupby("cluster_label")
            .agg(
                accuracy=("accuracy", "mean"),
                avg_time_taken=("avg_time_taken", "mean"),
                avg_attempts=("avg_attempts", "mean"),
                mastery_score=("mastery_score", "mean"),
                total_questions=("total_questions", "mean"),
            )
            .reset_index()
        )

        # ---- Ranking-based logic (IMPORTANT FIX) ----
        cluster_stats["accuracy_rank"] = cluster_stats["accuracy"].rank(
            ascending=False, method="first"
        )
        cluster_stats["time_rank"] = cluster_stats["avg_time_taken"].rank(
            ascending=True, method="first"
        )

        def assign_cluster_name(row):
            if row["accuracy_rank"] == 1 and row["time_rank"] == 1:
                return "High Performers"
            elif row["accuracy_rank"] == cluster_stats["accuracy_rank"].max():
                return "Struggling Learners"
            elif row["avg_time_taken"] > cluster_stats["avg_time_taken"].median():
                return "Careful Thinkers"
            else:
                return "Average Learners"

        cluster_stats["cluster_name"] = cluster_stats.apply(assign_cluster_name, axis=1)

        # Map names back to individual students
        label_map = dict(
            zip(cluster_stats["cluster_label"], cluster_stats["cluster_name"])
        )
        student_data["cluster_name"] = student_data["cluster_label"].map(label_map)

        return student_data, cluster_stats

    # -------------------------------------------------
    # 3. VISUALIZE CLUSTERS (PCA)
    # -------------------------------------------------
    def visualize_clusters(self, student_data: pd.DataFrame, features: pd.DataFrame):
        """
        PCA-based visualization of student clusters.
        """

        scaled_features = self.scaler.transform(features)

        pca = PCA(n_components=2, random_state=self.random_state)
        reduced = pca.fit_transform(scaled_features)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=student_data["cluster_label"],
            cmap="viridis",
            alpha=0.7,
        )

        plt.colorbar(scatter)
        plt.title("Student Clustering Visualization")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")

        # Annotate each cluster ONCE
        for cluster_id in sorted(student_data["cluster_label"].unique()):
            mask = student_data["cluster_label"] == cluster_id
            center = reduced[mask].mean(axis=0)
            label = student_data.loc[mask, "cluster_name"].iloc[0]

            plt.text(
                center[0],
                center[1],
                label,
                fontsize=11,
                fontweight="bold",
                ha="center",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
            )

        plt.tight_layout()
        plt.show()

    # -------------------------------------------------
    # 4. EVALUATE CLUSTER QUALITY
    # -------------------------------------------------
    def evaluate_clusters(self, features: pd.DataFrame) -> float:
        """
        Compute silhouette score for clustering quality.
        """
        scaled_features = self.scaler.fit_transform(features)
        labels = self.kmeans.fit_predict(scaled_features)
        return silhouette_score(scaled_features, labels)
