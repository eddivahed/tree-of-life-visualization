# src/tree_of_life/visualization/integrated_renderer.py

import numpy as np
from vispy import scene, app, visuals
from vispy.scene import ViewBox, PanZoomCamera
from PIL import Image
import logging
from pathlib import Path
from tqdm import tqdm
import cv2  # Add this import



class LoadingSpinner:
    def __init__(self, canvas):
        self.canvas = canvas
        self.loading_text = scene.visuals.Text(
            "Loading Images... 0%",
            parent=self.canvas.scene,
            color="white",
            font_size=24,
            pos=(self.canvas.size[0] / 2, self.canvas.size[1] / 2),
            anchor_x="center",
            anchor_y="center",
        )
        self.progress = 0

    def update(self, progress):
        self.progress = progress
        self.loading_text.text = f"Loading Images... {int(progress)}%"
        self.canvas.update()

    def hide(self):
        self.loading_text.parent = None
        self.canvas.update()


class IntegratedVisualizer:
    def __init__(self, navigation_system=None, width=1920, height=1080):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Store navigation system
        self.nav_system = navigation_system

        # Create canvas
        self.canvas = scene.SceneCanvas(
            keys="interactive",
            size=(width, height),
            show=True,
            title="Tree of Life Integrated Visualization",
            bgcolor="black",
        )

        # Create grid layout
        self.grid = self.canvas.central_widget.add_grid()

        # Main view for clustered images
        self.main_view = self.grid.add_view(row=0, col=0)
        self.main_view.camera = PanZoomCamera(aspect=1)

        # GAN view for generated images
        if self.nav_system:
            self.gan_view = self.grid.add_view(row=0, col=1)
            self.gan_view.camera = PanZoomCamera(aspect=1)
            self.gan_view.camera.set_range(x=(-1, 1), y=(-1, 1))
            self.generated_images = []
            self.grid_size = 4  # 4x4 grid

        # Initialize components
        self.image_nodes = []
        self.current_scale = 0.08
        self.movement_speed = 0.5
        self.loading_spinner = LoadingSpinner(self.canvas)

        # Setup interactions
        self.canvas.events.key_press.connect(self._on_key_press)
        self.canvas.events.mouse_wheel.connect(self._on_mouse_wheel)
        if self.nav_system:
            self.canvas.events.mouse_move.connect(self._on_mouse_move)
        self._setup_views()

    def _setup_views(self):
        """Set up the initial view parameters"""
        self.main_view.camera.set_range(x=(-2, 2), y=(-2, 2), margin=0.1)

    def _on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming"""
        if event.delta[1] > 0:
            self._adjust_image_scale(1.05)
        else:
            self._adjust_image_scale(0.95)

    def _adjust_image_scale(self, factor):
        """Adjust the scale of all images"""
        self.current_scale *= factor
        for node in self.image_nodes:
            if hasattr(node, "transform"):
                pos = node.transform.translate[:2]
                node.transform.scale = (self.current_scale, self.current_scale)
                node.transform.translate = (pos[0], pos[1], 0)

    def _on_key_press(self, event):
        """Handle keyboard interactions"""
        if event.key == "R":
            self._setup_views()
        elif event.key == "+":
            self._adjust_image_scale(1.1)
        elif event.key == "-":
            self._adjust_image_scale(0.9)
        elif event.key in ["Left", "Right", "Up", "Down"]:
            self._move_camera(event.key.lower())

    def _move_camera(self, direction):
        """Move the camera in the specified direction"""
        if direction == "left":
            self.main_view.camera.pan((self.movement_speed, 0, 0))
        elif direction == "right":
            self.main_view.camera.pan((-self.movement_speed, 0, 0))
        elif direction == "up":
            self.main_view.camera.pan((0, -self.movement_speed, 0))
        elif direction == "down":
            self.main_view.camera.pan((0, self.movement_speed, 0))

    def _on_mouse_move(self, event):
        """Handle mouse movement for GAN interaction"""
        if self.nav_system and event.button == 1 and event.is_dragging:
            pos = self.main_view.camera.transform.imap(event.pos)
            self._update_gan_view((pos[0], pos[1], 0))

    def _update_gan_view(self, position):
        """Update GAN view with generated patterns"""
        if not self.nav_system:
            return
            
        variations = []
        grid_size = self.grid_size
        
        # Generate grid of variations
        for i in range(grid_size):
            for j in range(grid_size):
                offset_x = (i - grid_size/2) * 0.2
                offset_y = (j - grid_size/2) * 0.2
                pos = (position[0] + offset_x, position[1] + offset_y, 0)
                pattern = self.nav_system.move_to(pos)
                variations.append((pattern, (i, j)))
        
        # Clear previous images
        for img in self.generated_images:
            if img.parent:
                img.parent = None
        self.generated_images.clear()
        
        # Create new image grid
        cell_size = 0.4
        for pattern, (i, j) in variations:
            try:
                img_data = self._pattern_to_image(pattern)
                
                image = scene.visuals.Image(
                    img_data,
                    parent=self.gan_view.scene,
                    method='subdivide'
                )
                
                x = -0.8 + i * cell_size
                y = -0.8 + j * cell_size
                
                image.transform = scene.STTransform(
                    translate=(x, y, 0),
                    scale=(cell_size * 0.9, cell_size * 0.9)
                )
                
                self.generated_images.append(image)
                
            except Exception as e:
                self.logger.error(f"Error creating pattern visualization: {e}")


    def _pattern_to_image(self, pattern):
        """Convert GAN pattern to displayable image with proper resizing"""
        # Normalize pattern
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        
        # Resize pattern to 64x64
        pattern_resized = cv2.resize(pattern, (64, 64), interpolation=cv2.INTER_CUBIC)
        
        # Create RGB image
        rgb = np.zeros((64, 64, 4))
        
        # Create color pattern
        x, y = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64))
        radius = np.sqrt(x**2 + y**2)
        gradient = np.exp(-radius * 3)
        
        # Create interesting color variations
        rgb[..., 0] = pattern_resized * gradient  # Red
        rgb[..., 1] = (1 - pattern_resized) * gradient  # Green
        rgb[..., 2] = np.abs(np.sin(pattern_resized * np.pi)) * gradient  # Blue
        rgb[..., 3] = np.ones((64, 64))  # Alpha channel
        
        # Add some noise for texture
        noise = np.random.rand(64, 64) * 0.1
        rgb[..., :3] += noise[..., np.newaxis]
        rgb = np.clip(rgb, 0, 1)
        
        return rgb

    def load_image(self, image_path, target_size=(64, 64)):
        """Load and process an image"""
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGBA")
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                return np.array(img)
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            return np.full((*target_size, 4), [255, 0, 0, 255], dtype=np.uint8)

    def set_data(self, points, image_paths):
        """Update visualization with clustered images"""
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
                    img_data, parent=self.main_view.scene, method="subdivide"
                )

                # Position image
                image.transform = scene.STTransform(
                    translate=(point[0], point[1], 0),
                    scale=(self.current_scale, self.current_scale),
                )

                self.image_nodes.append(image)

                if idx % 10 == 0:
                    app.process_events()

            except Exception as e:
                self.logger.error(f"Error creating image node: {e}")

        # Hide loading spinner
        self.loading_spinner.hide()

        # Initial GAN view if available
        if self.nav_system:
            self._update_gan_view((0, 0, 0))

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
        if self.nav_system:
            info_text += "\n        - Left click & drag: Generate new patterns"

        text = scene.visuals.Text(
            info_text,
            parent=self.canvas.scene,
            color="white",
            pos=(10, 10),
            font_size=10,
        )


def main():
    """Test the visualization"""
    viz = IntegratedVisualizer()

    try:
        # Get correct paths
        project_root = Path(__file__).parent.parent.parent.parent
        data_dir = project_root / "data"

        # Load embeddings and paths
        embeddings = np.load(data_dir / "embeddings.npy")
        with open(data_dir / "valid_paths.txt", "r") as f:
            paths = [Path(line.strip()) for line in f]

        viz.set_data(embeddings, paths)
        app.run()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the clustering first using:")
        print("poetry run python -m src.tree_of_life.main")


if __name__ == "__main__":
    main()
