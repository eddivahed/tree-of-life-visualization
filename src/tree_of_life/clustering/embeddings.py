# src/tree_of_life/clustering/embeddings.py
import numpy as np
from sklearn.preprocessing import StandardScaler
from umap.umap_ import UMAP  # Make sure this import is correct

class ImageEmbedder:
    """Handles dimension reduction and clustering of image features"""
    
    def __init__(self, n_components=3, n_neighbors=15, min_dist=0.1):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.scaler = StandardScaler()
        # Fixed this line: creating a UMAP instance
        self.umap = UMAP(            # Changed from umap() to UMAP()
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )
        
    def fit_transform(self, features):
        """Transform image features to lower-dimensional space"""
        # Normalize features
        scaled_features = self.scaler.fit_transform(features)
        
        # Apply UMAP
        embeddings = self.umap.fit_transform(scaled_features)
        return embeddings