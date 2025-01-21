# src/tree_of_life/visualization/grid_renderer.py

import numpy as np
from vispy import scene, app, visuals
from vispy.scene import ViewBox, PanZoomCamera
from PIL import Image
import logging
from pathlib import Path
from tqdm import tqdm

class LoadingSpinner:
    def __init__(self, canvas):
        self.canvas = canvas
        self.loading_text = scene.visuals.Text(
            'Loading Images... 0%',
            parent=self.canvas.scene,
            color='white',
            font_size=24,
            pos=(self.canvas.size[0]/2, self.canvas.size[1]/2),
            anchor_x='center',
            anchor_y='center'
        )
        self.progress = 0
        
    def update(self, progress):
        self.progress = progress
        self.loading_text.text = f'Loading Images... {int(progress)}%'
        self.canvas.update()
        
    def hide(self):
        self.loading_text.parent = None
        self.canvas.update()

class GridVisualizer:
    def __init__(self, width=1800, height=1600):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create canvas
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            size=(width, height),
            show=True,
            title='Tree of Life Grid Visualization',
            bgcolor='black'
        )

        # Create view with PanZoomCamera
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = PanZoomCamera(aspect=1)
        
        # Initialize parameters
        self.image_nodes = []
        self.current_scale = 0.08
        self.movement_speed = 0.5  # Increased for more noticeable movement
        self.loading_spinner = LoadingSpinner(self.canvas)
        
        # Setup interactions
        self.canvas.events.key_press.connect(self._on_key_press)
        self.canvas.events.mouse_wheel.connect(self._on_mouse_wheel)
        self._setup_grid_view()

    def _setup_grid_view(self):
        """Set up the initial view parameters"""
        self.view.camera.set_range(x=(-2, 2), y=(-2, 2), margin=0.1)

    def _on_key_press(self, event):
        """Handle keyboard interactions"""
        if event.key == 'r':
            self.view.camera.set_range(x=(-2, 2), y=(-2, 2), margin=0.1)
        elif event.key == 'o':
            self._adjust_image_scale(1.1)
        elif event.key == 'p':
            self._adjust_image_scale(0.9)
        elif event.key == 'd':
            self._move_camera('left')
        elif event.key == 'a':
            self._move_camera('right')
        elif event.key == 'w':
            self._move_camera('up')
        elif event.key == 's':
            self._move_camera('down')

    def _move_camera(self, direction):
        """Move the camera in the specified direction"""
        # Get current camera range
        x_min, x_max = self.view.camera.get_range()[0]
        y_min, y_max = self.view.camera.get_range()[1]
        
        # Calculate movement amount
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_move = x_range * self.movement_speed
        y_move = y_range * self.movement_speed
        
        # Apply movement based on direction
        if direction == 'left':
            x_min -= x_move
            x_max -= x_move
        elif direction == 'right':
            x_min += x_move
            x_max += x_move
        elif direction == 'up':
            y_min += y_move
            y_max += y_move
        elif direction == 'down':
            y_min -= y_move
            y_max -= y_move
            
        # Update camera range
        self.view.camera.set_range(x=(x_min, x_max), y=(y_min, y_max))

    def _on_mouse_wheel(self, event):
        """Handle mouse wheel for smoother zooming"""
        if event.delta[1] > 0:
            self._adjust_image_scale(1.05)
        else:
            self._adjust_image_scale(0.95)

    def _adjust_image_scale(self, factor):
        """Adjust the scale of all images with smooth transition"""
        self.current_scale *= factor
        for node in self.image_nodes:
            if hasattr(node, 'transform'):
                pos = node.transform.translate[:2]
                node.transform.scale = (self.current_scale, self.current_scale)
                node.transform.translate = (pos[0], pos[1], 0)

    def load_image(self, image_path, target_size=(64, 64)):
        """Load and process an image"""
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGBA')
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                return np.array(img)
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            return np.full((*target_size, 4), [255, 0, 0, 255], dtype=np.uint8)

    def set_data(self, points, image_paths):
        """Update visualization with grid-distributed images"""
        total_images = len(image_paths)
        
        # Clear existing images
        for node in self.image_nodes:
            if node.parent:
                node.parent = None
        self.image_nodes.clear()

        # Center the visualization
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        points = points - np.array([center_x, center_y])

        # Create image sprites with loading progress
        for idx, (point, img_path) in enumerate(zip(points, image_paths)):
            try:
                # Update loading progress
                progress = (idx + 1) / total_images * 100
                self.loading_spinner.update(progress)
                
                # Load and process image
                img_data = self.load_image(str(img_path))
                
                # Create image sprite
                image = scene.visuals.Image(
                    img_data,
                    parent=self.view.scene,
                    method='subdivide',
                    cmap='grays'
                )
                
                # Position image
                image.transform = scene.STTransform(
                    translate=(point[0], point[1], 0),
                    scale=(self.current_scale, self.current_scale)
                )
                
                self.image_nodes.append(image)
                
                # Update UI periodically
                if idx % 10 == 0:
                    app.process_events()
                
            except Exception as e:
                self.logger.error(f"Error creating image node: {e}")

        # Hide loading spinner
        self.loading_spinner.hide()

        # Center the view
        self.view.camera.set_range(x=(-2, 2), y=(-2, 2), margin=0.1)

    def add_controls_info(self):
        """Add control information overlay"""
        info_text = """
        Controls:
        - Arrow Keys: Navigate
        - Mouse drag: Pan view
        - Mouse wheel: Zoom
        - R: Reset view
        - +/-: Adjust image size
        """
        
        text = scene.visuals.Text(
            info_text,
            parent=self.canvas.scene,
            color='white',
            pos=(10, 10),
            font_size=10
        )

def main():
    viz = GridVisualizer()
    viz.add_controls_info()
    
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