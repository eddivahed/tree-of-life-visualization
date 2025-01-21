# src/tree_of_life/data/downloader.py

import requests
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import logging
import time

class OpenImagesDownloader:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.csv_dir = self.data_dir / "csv"
        
        # Create directories
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
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
                self.logger.info(f"Downloading {name} metadata...")
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    self.logger.info(f"Successfully downloaded {name}")
                except Exception as e:
                    self.logger.error(f"Error downloading {name}: {e}")
                    raise

    def get_nature_images(self, max_images=2000):
        """Get nature/tree related image URLs"""
        self.logger.info("Reading class descriptions...")
        classes_df = pd.read_csv(self.csv_dir / "class_descriptions.csv", 
                               names=['LabelName', 'DisplayName'])
        
        self.logger.info("Filtering nature-related classes...")
        nature_classes = classes_df[
            classes_df['DisplayName'].str.lower().str.contains('tree|plant|flower|nature|leaf')
        ]['LabelName'].tolist()
        
        self.logger.info(f"Found {len(nature_classes)} nature-related classes")
        
        self.logger.info("Reading annotations...")
        boxes_df = pd.read_csv(self.csv_dir / "train_boxes.csv")
        
        self.logger.info("Filtering images with nature classes...")
        nature_images = boxes_df[
            boxes_df['LabelName'].isin(nature_classes)
        ]['ImageID'].unique()
        
        self.logger.info(f"Found {len(nature_images)} images with nature content")
        
        self.logger.info("Reading image URLs...")
        images_df = pd.read_csv(self.csv_dir / "train_images.csv")
        nature_urls = images_df[
            images_df['ImageID'].isin(nature_images)
        ]['OriginalURL'].tolist()[:max_images]
        
        self.logger.info(f"Selected {len(nature_urls)} URLs for download")
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
                    # Add a small delay to prevent rate limiting
                    time.sleep(0.1)
                except Exception as e:
                    self.logger.error(f"Error downloading image {idx}: {e}")
            else:
                self.logger.debug(f"Image {idx} already exists, skipping")