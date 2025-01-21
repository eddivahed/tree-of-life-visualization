# src/tree_of_life/clustering/processor.py

import numpy as np
import cv2
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from umap.umap_ import UMAP
import logging

class FeatureExtractor:
    def __init__(self):
        self.target_size = (224, 224)  # Standard size for processing
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _compute_color_histogram(self, image):
        """Compute color histogram features."""
        # Convert to HSV color space for better color representation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Compute histograms for each channel
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        # Normalize histograms
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()
        
        return np.concatenate([h_hist, s_hist, v_hist])

    def _compute_texture_features(self, image):
        """Compute texture features using Gabor filters."""
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Initialize Gabor features
        gabor_features = []
        
        # Define Gabor filter parameters
        orientations = [0, 45, 90, 135]
        scales = [1, 2]
        
        for theta in orientations:
            for scale in scales:
                kernel = cv2.getGaborKernel(
                    ksize=(31, 31),
                    sigma=scale,
                    theta=theta * np.pi / 180,
                    lambd=10,
                    gamma=0.5,
                    psi=0
                )
                
                # Apply filter and compute statistics
                filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
                features = [
                    np.mean(filtered),
                    np.std(filtered),
                    np.max(filtered),
                    np.min(filtered)
                ]
                gabor_features.extend(features)
        
        return np.array(gabor_features)

    def extract_features(self, image_path):
        """Extract combined features from an image."""
        try:
            # Read and resize image
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return None
                
            image = cv2.resize(image, self.target_size)
            
            # Extract features
            color_features = self._compute_color_histogram(image)
            texture_features = self._compute_texture_features(image)
            
            # Combine features
            combined_features = np.concatenate([color_features, texture_features])
            return combined_features
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            return None

class ImageClusterer:
    def __init__(self, n_components=3, n_neighbors=15, min_dist=0.1):
        self.scaler = StandardScaler()
        self.umap = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )
        self.logger = logging.getLogger(__name__)

    def process_directory(self, image_dir):
        """Process all images in a directory and return their embeddings."""
        image_dir = Path(image_dir)
        feature_extractor = FeatureExtractor()
        
        # Collect features for all images
        features = []
        valid_paths = []
        
        for img_path in image_dir.glob('*.jpg'):
            feature_vector = feature_extractor.extract_features(img_path)
            if feature_vector is not None:
                features.append(feature_vector)
                valid_paths.append(img_path)
        
        if not features:
            self.logger.error("No valid features extracted from images")
            return None, []
            
        # Convert to numpy array
        features = np.array(features)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Generate embeddings
        embeddings = self.umap.fit_transform(scaled_features)
        
        return embeddings, valid_paths

def main():
    # Test the implementation
    clusterer = ImageClusterer()
    embeddings, paths = clusterer.process_directory("data/images")
    
    if embeddings is not None:
        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding shape: {embeddings.shape}")
        
        # Save embeddings for visualization
        np.save("data/embeddings.npy", embeddings)
        
        # Save paths for reference
        with open("data/valid_paths.txt", "w") as f:
            for path in paths:
                f.write(f"{path}\n")

if __name__ == "__main__":
    main()