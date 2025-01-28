# Tree of Life Visualization - Product Requirements Document

## 1. Project Overview

A Python-based application that downloads nature/tree images, processes them using machine learning techniques, and creates an interactive 3D visualization of image clusters.

## 2. Technical Architecture

### 2.1 Environment Setup
```bash
tree-of-life-project/
├── pyproject.toml      # Poetry dependency management
├── src/
│   └── tree_of_life/
│       ├── data/       # Data handling modules
│       ├── clustering/ # ML processing modules
│       └── viz/        # Visualization modules
└── notebooks/         # Development/testing notebooks
```

### 2.2 Core Dependencies

- **Environment Management**
  - Poetry (dependency management)
  - Python 3.9+

- **Data Processing**
  - NumPy
  - Pandas
  - OpenCV
  - Pillow

- **Machine Learning**
  - UMAP
  - scikit-learn

- **Visualization**
  - Vispy

## 3. Implementation Phases

### Phase 1: Data Pipeline

#### 3.1.1 Image Collection
- Download metadata from Open Images Dataset V7
- Filter for nature/tree related images
- Download and store filtered images
- Implement rate limiting and error handling

Implementation Priority:
1. Basic metadata download
2. Image filtering logic
3. Batch download system
4. Error handling and retry logic

#### 3.1.2 Image Processing
- Load and preprocess images
- Extract color features
- Extract texture features
- Optional: Extract CNN features

### Phase 2: Clustering System

#### 3.2.1 Feature Processing
- Normalize extracted features
- Implement dimensionality reduction (UMAP)
- Create cluster embeddings

#### 3.2.2 Clustering Logic
- Implement real-time cluster updates
- Add cluster metadata tracking
- Create cluster relationship mapping

### Phase 3: Visualization

#### 3.3.1 Basic Visualization
- 3D point cloud representation
- Basic camera controls
- Color mapping system

#### 3.3.2 Interactive Features
- Zoom/pan/rotate controls
- Cluster selection
- Image preview on hover
- Smooth transitions

## 4. Detailed Implementation Plan

### 4.1 Data Pipeline Implementation

```python
# data/downloader.py
# src/tree_of_life/data/downloader.py
import requests
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd

class OpenImagesDownloader:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.csv_dir = self.data_dir / "csv"
        
        # Create directories
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        
        # URLs from Google Storage
        self.metadata_urls = {
            'train_boxes': 'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv',
            'class_descriptions': 'https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv',
            'train_images': 'https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv'
        }

    def download_metadata(self):
        """Download metadata files"""
        for name, url in self.metadata_urls.items():
            output_path = self.csv_dir / f"{name}.csv"
            if not output_path.exists():
                print(f"Downloading {name} metadata...")
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Successfully downloaded {name}")
                except Exception as e:
                    print(f"Error downloading {name}: {e}")

    def get_nature_images(self, max_images=100):
        """Get nature/tree related image URLs"""
        # Read class descriptions
        classes_df = pd.read_csv(self.csv_dir / "class_descriptions.csv", 
                               names=['LabelName', 'DisplayName'])
        
        # Filter nature-related classes
        nature_classes = classes_df[
            classes_df['DisplayName'].str.lower().str.contains('tree|plant|flower|nature|leaf')
        ]['LabelName'].tolist()
        
        # Read train boxes
        boxes_df = pd.read_csv(self.csv_dir / "train_boxes.csv")
        
        # Get image IDs with nature classes
        nature_images = boxes_df[
            boxes_df['LabelName'].isin(nature_classes)
        ]['ImageID'].unique()
        
        # Read image URLs
        images_df = pd.read_csv(self.csv_dir / "train_images.csv")
        nature_urls = images_df[
            images_df['ImageID'].isin(nature_images)
        ]['OriginalURL'].tolist()[:max_images]
        
        return nature_urls

    def download_images(self, urls):
        """Download images from URLs"""
        for idx, url in enumerate(tqdm(urls, desc="Downloading images")):
            # Generate filename from URL
            filename = f"image_{idx:04d}.jpg"
            output_path = self.image_dir / filename
            
            if not output_path.exists():
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                except Exception as e:
                    print(f"Error downloading image {idx}: {e}")
```

### 4.2 Clustering Implementation

```python
# clustering/embeddings.py
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
```

### 4.3 Visualization Implementation

```python
# src/tree_of_life/visualization/renderer.py
import numpy as np
from vispy import scene, app
from vispy.scene import ViewBox, PanZoomCamera
from PIL import Image

class ImageVisualizer:
    def __init__(self, width=1024, height=768):
        # Create canvas with a scene
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            size=(width, height),
            show=True,
            title='Tree of Life Visualization'
        )

        # Create view with PanZoomCamera instead of TurntableCamera
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = PanZoomCamera(aspect=1)
        self.view.camera.rect = (-1, -1, 2, 2)  # Set initial view

        # Container for image nodes
        self.image_nodes = []

    def load_image(self, image_path, target_size=(64, 64)):
        """Load and resize an image, returning it as a numpy array"""
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGBA')  # Convert to RGBA for transparency
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                return np.array(img)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a colored placeholder if image loading fails
            return np.full((*target_size, 4), [255, 0, 0, 255], dtype=np.uint8)

    def set_data(self, points, image_paths):
        """Update visualization with images at specified points"""
        # Clear existing images
        for node in self.image_nodes:
            if node.parent:
                node.parent = None
        self.image_nodes.clear()

        # Create image sprites for each point
        for point, img_path in zip(points, image_paths):
            try:
                # Load and process image
                img_data = self.load_image(img_path)
                
                # Create image sprite
                image = scene.visuals.Image(
                    img_data,
                    parent=self.view.scene,
                    method='subdivide'  # Better quality rendering
                )
                
                # Position and scale image
                scale = 0.1  # Adjust this value to change image size
                image.transform = scene.STTransform(
                    translate=(point[0] - scale/2, point[1] - scale/2),
                    scale=(scale, scale)
                )
                
                self.image_nodes.append(image)
                
            except Exception as e:
                print(f"Error creating image node: {e}")

        # Update view
        self.view.camera.set_range()

def main():
    # Test the visualization
    viz = ImageVisualizer()
    
    # Create some test points
    points = np.random.randn(5, 3) * 0.5
    
    # Use placeholder images or actual image paths
    image_paths = [
        "path/to/your/images/image1.jpg",
        "path/to/your/images/image2.jpg",
        # ... add more image paths
    ]
    
    viz.set_data(points, image_paths)
    app.run()

if __name__ == "__main__":
    main()
```

## 5. Development Milestones

### Milestone 1: Basic Setup
- [x] Project structure
- [x] Dependency management
- [x] Basic downloader

### Milestone 2: Core Functionality
- [ ] Complete image download pipeline
- [ ] Basic feature extraction
- [ ] Simple UMAP clustering

### Milestone 3: Visualization
- [ ] Basic 3D visualization
- [ ] Camera controls
- [ ] Point cloud rendering

### Milestone 4: Advanced Features
- [ ] Interactive selection
- [ ] Image previews
- [ ] Cluster transitions

## 6. Testing Strategy

### Unit Tests
- Data pipeline components
- Feature extraction methods
- Clustering algorithms
- Visualization modules

### Integration Tests
- End-to-end pipeline testing
- Data flow validation
- Error handling verification

### Performance Testing
- Visualization frame rate
- Memory usage monitoring
- Data processing speed
- Clustering efficiency

### User Interaction Testing
- Control responsiveness
- UI element functionality
- Error message clarity

## 7. Next Steps

### Immediate Implementation Priorities

1. Complete download pipeline:
```python
class OpenImagesDownloader:
    def download_batch(self, image_ids):
        """
        Download a batch of images with:
        - Progress tracking
        - Error handling
        - Rate limiting
        - Retry logic
        """
        pass
```

2. Feature extraction implementation:
```python
class FeatureExtractor:
    def process_image(self, image_path):
        """
        Extract features with:
        - Color histograms
        - Texture patterns
        - Shape descriptors
        """
        pass
```

### Future Enhancements
1. GPU acceleration for feature extraction
2. Advanced clustering algorithms
3. Enhanced visualization effects
4. Real-time data updates

## 8. Documentation

### Required Documentation
- Installation guide
- API documentation
- Usage examples
- Performance optimization guide
- Troubleshooting guide

### Code Documentation Standards
- Docstring format: Google style
- Type hints for all functions
- README files for each module
- Example notebooks

## 9. Performance Considerations

### Optimization Targets
- Image processing pipeline
- UMAP computation
- Visualization rendering
- Memory management

### Scalability Concerns
- Large dataset handling
- Batch processing
- Memory-efficient operations
- Caching strategies
