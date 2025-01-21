# src/tree_of_life/visualization/renderer.py
import numpy as np
from vispy import scene, app, visuals
from vispy.scene import ViewBox, TurntableCamera
from PIL import Image
import logging
from pathlib import Path

class ImprovedVisualizer:
    def __init__(self, width=1200, height=800):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create canvas
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            size=(width, height),
            show=True,
            title='Tree of Life Visualization'
        )

        # Create 3D view
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = TurntableCamera(
            elevation=30,
            azimuth=45,
            distance=2.0,
            center=(0, 0, 0)
        )

        # Enable interactive features
        self.canvas.events.key_press.connect(self._on_key_press)
        self.canvas.events.mouse_wheel.connect(self._on_mouse_wheel)
        
        # Container for image nodes
        self.image_nodes = []
        self.current_scale = 0.15  # Initial image scale

    def _on_key_press(self, event):
        """Handle keyboard interactions"""
        if event.key == 'R':
            self.view.camera.reset()
        elif event.key == '+':
            self._adjust_image_scale(1.1)
        elif event.key == '-':
            self._adjust_image_scale(0.9)

    def _on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming"""
        self._adjust_image_scale(1.1 if event.delta[1] > 0 else 0.9)

    def _adjust_image_scale(self, factor):
        """Adjust the scale of all images"""
        self.current_scale *= factor
        for node in self.image_nodes:
            if hasattr(node, 'transform'):
                pos = node.transform.translate[:2]
                node.transform.scale = (self.current_scale, self.current_scale)
                # Adjust position to maintain center point
                node.transform.translate = (
                    pos[0] - self.current_scale/2,
                    pos[1] - self.current_scale/2,
                    0
                )

    def load_image(self, image_path, target_size=(128, 128)):
        """Load and process an image for visualization"""
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGBA')
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                return np.array(img)
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            # Return a colored placeholder
            return np.full((*target_size, 4), [255, 0, 0, 255], dtype=np.uint8)

    def set_data(self, points, image_paths):
        """Update visualization with clustered images"""
        # Clear existing images
        for node in self.image_nodes:
            if node.parent:
                node.parent = None
        self.image_nodes.clear()

        # Normalize points to fit in view
        points = np.array(points)
        points = (points - points.mean(axis=0)) / (points.std(axis=0) * 2)

        # Create image sprites for each point
        for point, img_path in zip(points, image_paths):
            try:
                # Load and process image
                img_data = self.load_image(str(img_path))
                
                # Create image sprite
                image = scene.visuals.Image(
                    img_data,
                    parent=self.view.scene,
                    method='subdivide'
                )
                
                # Position image
                image.transform = scene.STTransform(
                    translate=(
                        point[0] - self.current_scale/2,
                        point[1] - self.current_scale/2,
                        point[2] if len(point) > 2 else 0
                    ),
                    scale=(self.current_scale, self.current_scale)
                )
                
                self.image_nodes.append(image)
                
            except Exception as e:
                self.logger.error(f"Error creating image node: {e}")

        # Update view
        self.view.camera.set_range()

    def add_controls_info(self):
        """Add control information to the visualization"""
        info_text = """
        Controls:
        - Mouse drag: Rotate view
        - Mouse wheel: Zoom
        - R: Reset view
        - +/-: Adjust image size
        """
        
        # Create text visual
        text = scene.visuals.Text(
            info_text,
            parent=self.canvas.scene,
            color='white',
            pos=(10, 10),
            font_size=10
        )

def main():
    # Test the visualization with some sample data
    viz = ImprovedVisualizer()
    viz.add_controls_info()
    
    # Load embeddings and paths if they exist
    try:
        embeddings = np.load("data/embeddings.npy")
        with open("data/valid_paths.txt", "r") as f:
            paths = [Path(line.strip()) for line in f]
            
        viz.set_data(embeddings, paths)
        app.run()
        
    except FileNotFoundError:
        print("Please run the clustering first to generate embeddings.")

if __name__ == "__main__":
    main()