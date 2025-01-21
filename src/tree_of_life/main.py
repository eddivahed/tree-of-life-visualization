# src/tree_of_life/main.py

import logging
from pathlib import Path
from .data.downloader import OpenImagesDownloader
from .clustering.enhanced_processor import EnhancedImageClusterer 
from .visualization.integrated_renderer import IntegratedVisualizer  # Updated import
from .gan_navigator.latent_explorer import NavigationSystem
from vispy import app
import numpy as np
import argparse

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def ensure_directories():
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "data"
    image_dir = data_dir / "images"
    csv_dir = data_dir / "csv"
    models_dir = data_dir / "models"  # New directory for GAN models
    
    for directory in [data_dir, image_dir, csv_dir, models_dir]:
        directory.mkdir(exist_ok=True)
        
    return base_dir, data_dir, image_dir, csv_dir, models_dir

def download_images(data_dir, image_dir, num_images=2000):
    """Download images from Open Images Dataset"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting download process for {num_images} images...")
    
    downloader = OpenImagesDownloader(str(data_dir))
    
    # Download metadata
    logger.info("Downloading metadata...")
    downloader.download_metadata()
    
    # Get and download nature images
    logger.info("Getting nature image URLs...")
    urls = downloader.get_nature_images(max_images=num_images)
    
    if urls:
        logger.info(f"Found {len(urls)} URLs. Downloading images...")
        downloader.download_images(urls)
    else:
        logger.error("No URLs found for downloading")
        return False
    
    # Verify downloads
    downloaded_images = list(image_dir.glob("*.jpg"))
    logger.info(f"Successfully downloaded {len(downloaded_images)} images")
    return len(downloaded_images) > 0

def initialize_gan(models_dir, embeddings=None):
    """Initialize or train GAN system"""
    logger = logging.getLogger(__name__)
    model_path = models_dir / "gan_model.pt"
    
    nav_system = NavigationSystem(model_path if model_path.exists() else None)
    
    # Train GAN if no model exists and we have embeddings
    if not model_path.exists() and embeddings is not None:
        logger.info("Training GAN model...")
        nav_system.train(embeddings)
        nav_system.save_model(model_path)
        logger.info("GAN model training complete")
    
    return nav_system

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tree of Life Visualization')
    parser.add_argument('--download', action='store_true', help='Force download new images')
    parser.add_argument('--num-images', type=int, default=2000, help='Number of images to download')
    parser.add_argument('--train-gan', action='store_true', help='Force GAN training')
    parser.add_argument('--no-gan', action='store_true', help='Disable GAN integration')
    args = parser.parse_args()

    logger = setup_logging()
    base_dir, data_dir, image_dir, csv_dir, models_dir = ensure_directories()
    
    # Check for existing images
    downloaded_images = list(image_dir.glob("*.jpg"))
    logger.info(f"Found {len(downloaded_images)} existing images")
    
    # Download images if needed
    if args.download or not downloaded_images:
        logger.info("Initiating image download process...")
        if not download_images(data_dir, image_dir, args.num_images):
            logger.error("Failed to download images. Please check your internet connection.")
            return
        downloaded_images = list(image_dir.glob("*.jpg"))
    
    if not downloaded_images:
        logger.error("No images available for processing.")
        return
    
    # Process images with enhanced clustering
    logger.info("Processing images with enhanced feature extraction...")
    clusterer = EnhancedImageClusterer(n_components=2)
    embeddings, valid_paths = clusterer.process_directory(image_dir)
    
    if embeddings is None or len(valid_paths) == 0:
        logger.error("No valid embeddings generated. Please check the image files.")
        return
    
    logger.info(f"Generated embeddings for {len(valid_paths)} images")
    
    # Save results
    np.save(data_dir / "embeddings.npy", embeddings)
    with open(data_dir / "valid_paths.txt", "w") as f:
        for path in valid_paths:
            f.write(f"{str(path)}\n")
    
    # Initialize GAN system if not disabled
    nav_system = None
    if not args.no_gan:
        if args.train_gan:
            # Remove existing model to force retraining
            model_path = models_dir / "gan_model.pt"
            if model_path.exists():
                model_path.unlink()
        
        nav_system = initialize_gan(models_dir, embeddings)
    
    # Visualize
    logger.info("Launching visualization...")
    viz = IntegratedVisualizer(nav_system) if nav_system else GridVisualizer()
    viz.add_controls_info()
    viz.set_data(embeddings, valid_paths)
    
    app.run()

if __name__ == "__main__":
    main()