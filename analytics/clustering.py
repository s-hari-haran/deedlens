"""
DeedLens Clustering
Groups similar properties using K-Means clustering.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class ClusterResult:
    """Result of clustering."""
    cluster_id: int
    doc_ids: List[str]
    centroid: np.ndarray
    size: int


class PropertyClusterer:
    """
    Clusters properties based on embeddings or features.
    """
    
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available. Install it first.")
        
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
    
    def cluster_by_embeddings(
        self,
        doc_ids: List[str],
        embeddings: List[np.ndarray]
    ) -> Dict[int, List[str]]:
        """
        Cluster documents by their embeddings.
        
        Args:
            doc_ids: List of document IDs
            embeddings: List of embedding vectors
        
        Returns:
            Dictionary mapping cluster ID to list of doc IDs
        """
        if len(embeddings) < self.n_clusters:
            self.n_clusters = max(2, len(embeddings) // 2)
        
        X = np.array(embeddings)
        
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        
        labels = self.model.fit_predict(X)
        
        # Group by cluster
        clusters = {}
        for doc_id, label in zip(doc_ids, labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(doc_id)
        
        return clusters
    
    def cluster_by_features(
        self,
        doc_ids: List[str],
        features: List[Dict]
    ) -> Dict[int, List[str]]:
        """
        Cluster documents by extracted features.
        
        Args:
            doc_ids: List of document IDs
            features: List of feature dictionaries
        
        Returns:
            Dictionary mapping cluster ID to list of doc IDs
        """
        # Extract numeric features
        feature_matrix = []
        
        for f in features:
            row = [
                f.get("area", 0),
                f.get("value", 0),
                f.get("year", 2020),
            ]
            feature_matrix.append(row)
        
        X = np.array(feature_matrix)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if len(X_scaled) < self.n_clusters:
            self.n_clusters = max(2, len(X_scaled) // 2)
        
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        
        labels = self.model.fit_predict(X_scaled)
        
        # Group by cluster
        clusters = {}
        for doc_id, label in zip(doc_ids, labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(doc_id)
        
        return clusters
    
    def get_cluster_info(self) -> List[ClusterResult]:
        """Get information about clusters."""
        if self.model is None:
            return []
        
        results = []
        for i in range(self.n_clusters):
            results.append(ClusterResult(
                cluster_id=i,
                doc_ids=[],  # Would need to be populated
                centroid=self.model.cluster_centers_[i],
                size=0
            ))
        
        return results


def cluster_documents(
    doc_ids: List[str],
    embeddings: List[np.ndarray],
    n_clusters: int = 5
) -> Dict:
    """
    Convenience function to cluster documents.
    
    Args:
        doc_ids: List of document IDs
        embeddings: List of embedding vectors
        n_clusters: Number of clusters
    
    Returns:
        Dictionary with cluster assignments
    """
    clusterer = PropertyClusterer(n_clusters=n_clusters)
    clusters = clusterer.cluster_by_embeddings(doc_ids, embeddings)
    
    return {
        "n_clusters": len(clusters),
        "clusters": {
            str(k): v for k, v in clusters.items()
        }
    }


if __name__ == "__main__":
    # Example usage
    doc_ids = [f"doc_{i}" for i in range(10)]
    embeddings = [np.random.rand(384) for _ in range(10)]
    
    result = cluster_documents(doc_ids, embeddings, n_clusters=3)
    
    print("=== Clusters ===")
    for cluster_id, docs in result["clusters"].items():
        print(f"Cluster {cluster_id}: {docs}")
