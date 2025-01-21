# src/tree_of_life/visualization/integrated_renderer.py

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

class IntegratedVisualizer:
    def __init__(self, navigation_system=None, width=1920, height=1080):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Store navigation system
        self.nav_system = navigation_system
        
        # Create canvas
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            size=(width, height),
            show=True,
            title='Tree of Life Integrated Visualization',
            bgcolor='black'
        )

        # Create grid layout
        self.grid = self.canvas.central_widget.add_grid()
        
        # Main view for clustered images
        self.main_view = self.grid.add_view(row=0, col=0)
        self.main_view.camera = PanZoomCamera(aspect=1)
        
        # Secondary view for GAN-generated images
        if self.nav_system:
            self.gan_view = self.grid.add_view(row=0, col=1)
            self.gan_view.camera = PanZoomCamera(aspect=1)
            self.gan_image = None  # Will store the GAN visualization
        
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
        if self.nav_system:
            self.gan_view.camera.set_range(x=(-1, 1), y=(-1, 1), margin=0.1)

    def _on_mouse_move(self, event):
        """Handle mouse movement for GAN interaction"""
        if self.nav_system and event.button == 1 and event.is_dragging:
            pos = self.main_view.camera.transform.imap(event.pos)
            self._update_gan_view((pos[0], pos[1], 0))

    def _update_gan_view(self, position):
        """Update GAN view with new generated image"""
        if not self.nav_system:
            return
            
        # Generate new positions using GAN
        new_pos = self.nav_system.move_to(position)
        
        # Create or update visualization
        if self.gan_image is None:
            self.gan_image = scene.visuals.Markers()
            self.gan_image.set_data(
                pos=np.array([new_pos]),
                symbol='square',
                edge_color='white',
                face_color='blue',
                size=20
            )
            self.gan_view.add(self.gan_image)
        else:
            self.gan_image.set_data(
                pos=np.array([new_pos]),
                symbol='square',
                edge_color='white',
                face_color='blue',
                size=20
            )

    def _on_key_press(self, event):
        """Handle keyboard interactions"""
        if event.key == 'R':
            self._setup_views()
        elif event.key == '+':
            self._adjust_image_scale(1.1)
        elif event.key == '-':
            self._adjust_image_scale(0.9)
        elif event.key in ['Left', 'Right', 'Up', 'Down']:
            self._move_camera(event.key.lower())

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
                    img_data,
                    parent=self.main_view.scene,
                    method='subdivide'
                )
                
                # Position image
                image.transform = scene.STTransform(
                    translate=(point[0], point[1], 0),
                    scale=(self.current_scale, self.current_scale)
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
            color='white',
            pos=(10, 10),
            font_size=10
        )
    
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
            if hasattr(node, 'transform'):
                pos = node.transform.translate[:2]
                node.transform.scale = (self.current_scale, self.current_scale)
                node.transform.translate = (pos[0], pos[1], 0)       

    def _move_camera(self, direction):
        """Move the camera in the specified direction"""
        if direction == 'left':
            self.main_view.camera.pan((self.movement_speed, 0, 0))
        elif direction == 'right':
            self.main_view.camera.pan((-self.movement_speed, 0, 0))
        elif direction == 'up':
            self.main_view.camera.pan((0, -self.movement_speed, 0))
        elif direction == 'down':
            self.main_view.camera.pan((0, self.movement_speed, 0))       
    