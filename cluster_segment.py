import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

def perform_clustering(df: pd.DataFrame, n_clusters: int = 3, return_data: bool = False):
    """
    Performs KMeans clustering on numerical features and visualizes the clusters using PCA.

    Args:
        df (pd.DataFrame): Input DataFrame with numerical features.
        n_clusters (int): Number of clusters for KMeans.
        return_data (bool): If True, returns the DataFrame with cluster labels and PCA.

    Returns:
        Plotly Figure of 2D PCA scatter plot, and optionally the DataFrame with cluster labels.
    """
    try:
        numeric_df = df.select_dtypes(include="number").dropna(axis=1)
        if numeric_df.shape[1] < 2:
            raise ValueError("At least two numeric columns are required for clustering.")

        # Standardize numeric features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df)

        # Apply KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Add cluster and PCA columns
        result_df = numeric_df.copy()
        result_df["Cluster"] = clusters

        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)
        result_df["PC1"] = components[:, 0]
        result_df["PC2"] = components[:, 1]

        # Plot clusters
        fig = px.scatter(
            result_df,
            x="PC1",
            y="PC2",
            color=result_df["Cluster"].astype(str),
            title=f"KMeans Clustering with {n_clusters} Clusters (PCA View)",
            labels={"color": "Cluster", "PC1": "Principal Component 1", "PC2": "Principal Component 2"},
            template="plotly_white"
        )

        if return_data:
            return fig, result_df
        return fig

    except Exception as e:
        print(f"[perform_clustering] Clustering failed: {e}")
        return None if not return_data else (None, None)
