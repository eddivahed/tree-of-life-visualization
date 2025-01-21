# src/tree_of_life/clustering/enhanced_processor.py

import numpy as np
import cv2
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from umap.umap_ import UMAP
import logging
from scipy.stats import skew, kurtosis
import warnings

class EnhancedFeatureExtractor:
    def __init__(self):
        self.target_size = (224, 224)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _safe_compute_stats(self, data):
        """Safely compute statistical measures with fallback values"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sk = float(skew(data))
                kt = float(kurtosis(data))
                
                # Check for invalid values
                if np.isnan(sk) or np.isinf(sk):
                    sk = 0.0
                if np.isnan(kt) or np.isinf(kt):
                    kt = 0.0
                    
                return sk, kt
        except:
            return 0.0, 0.0

    def _compute_color_features(self, image):
        """Extract comprehensive color features with robust error handling"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            features = []
            
            # Compute color histograms in multiple color spaces
            color_spaces = [
                ('bgr', image, [256, 256, 256]), 
                ('hsv', hsv, [180, 256, 256]),
                ('lab', lab, [256, 256, 256])
            ]
            
            for name, color_img, ranges in color_spaces:
                for channel in range(3):
                    # Compute histogram
                    hist = cv2.calcHist([color_img], [channel], None, [16], [0, ranges[channel]])
                    hist = cv2.normalize(hist, hist).flatten()
                    
                    # Compute basic statistics
                    channel_data = color_img[:,:,channel].astype(float)
                    mean = np.mean(channel_data)
                    std = np.std(channel_data)
                    
                    # Safely compute higher order statistics
                    sk, kt = self._safe_compute_stats(channel_data.flatten())
                    
                    features.extend([mean, std, sk, kt])
                    features.extend(hist)
            
            # Compute dominant colors more efficiently
            pixels = image.reshape(-1, 3).astype(np.float32)
            
            # Use k-means with a smaller sample for efficiency
            sample_size = min(10000, pixels.shape[0])
            pixels_sample = pixels[np.random.choice(pixels.shape[0], sample_size, replace=False)]
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(pixels_sample, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Add dominant colors to features
            features.extend(centers.flatten())
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Error in color feature extraction: {str(e)}")
            # Return a zero vector of appropriate length if processing fails
            return np.zeros(404, dtype=np.float32)  # Adjust size based on your feature vector length

    def _compute_texture_features(self, image):
        """Extract texture features with robust error handling"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            features = []
            
            # Simplified Gabor filter bank
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
                    filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                    features.extend([
                        np.mean(filtered),
                        np.std(filtered)
                    ])
            
            # Simple edge detection features
            edges = cv2.Canny(gray, 100, 200)
            features.extend([
                np.mean(edges),
                np.std(edges),
                np.sum(edges > 0) / edges.size  # Edge density
            ])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Error in texture feature extraction: {str(e)}")
            return np.zeros(19, dtype=np.float32)  # Adjust size based on your feature vector length

    def extract_features(self, image_path):
        """Extract combined visual features from an image with robust error handling"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return None
                
            image = cv2.resize(image, self.target_size)
            
            # Extract features
            color_features = self._compute_color_features(image)
            texture_features = self._compute_texture_features(image)
            
            # Combine features
            combined_features = np.concatenate([
                color_features,
                texture_features
            ])
            
            # Ensure no invalid values
            combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            return combined_features
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            return None

class EnhancedImageClusterer:
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1):
        self.scaler = RobustScaler()  # Using RobustScaler instead of StandardScaler
        self.umap = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
            metric='euclidean'
        )
        self.logger = logging.getLogger(__name__)

    def process_directory(self, image_dir):
        """Process all images and return their embeddings"""
        image_dir = Path(image_dir)
        feature_extractor = EnhancedFeatureExtractor()
        
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
            
        features = np.array(features)
        
        # Remove any remaining invalid values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Generate embeddings
        embeddings = self.umap.fit_transform(scaled_features)
        
        return embeddings, valid_paths

def main():
    # Test the implementation
    clusterer = EnhancedImageClusterer()
    embeddings, paths = clusterer.process_directory("data/images")
    
    if embeddings is not None:
        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding shape: {embeddings.shape}")

if __name__ == "__main__":
    main()