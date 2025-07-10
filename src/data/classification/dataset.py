# In: src/data/classification/dataset.py

import torch
import os
import json
import logging
from PIL import Image, UnidentifiedImageError

# Setup a logger for better debugging
logger = logging.getLogger(__name__)

class OralClassificationDataset(torch.utils.data.Dataset):
    """
    A unified dataset for classification that supports both directory-based
    (ImageFolder-style) and COCO-style JSON data sources.
    """
    def __init__(self, data_path: str, transform=None):
        super().__init__()
        self.transform = transform
        self.classes = []
        # This will hold our unified list of samples: [(image_path, class_id), ...]
        self.samples = []
        
        # This will map class names to integer IDs, e.g., {'CANCER': 0, 'NON CANCER': 1}
        self.class_to_idx = {}
        
        # Load data using the dispatcher logic
        self._load_data_source(data_path)
        
        
        
        if not self.samples:
            raise ValueError(f"No valid data found at path: {data_path}")
            
        logger.info(f"Dataset initialized from {data_path}. Found {len(self.samples)} samples.")

    def _load_data_source(self, path: str):
        """
        Detects the type of data source (directory or JSON file)
        and calls the appropriate loading method.
        """
        if not os.path.exists(path):
            logger.error(f"Data source path does not exist: {path}")
            return

        if os.path.isdir(path):
            logger.info(f"Path is a directory. Loading with directory-based strategy.")
            self._load_from_directory(path)
        elif path.endswith('.json'):
            logger.info(f"Path is a JSON file. Loading with COCO-style strategy.")
            self._load_from_json(path)
        else:
            logger.warning(f"Unrecognized data source type: {path}. Cannot load data.")

    def _load_from_directory(self, dir_path: str):
        """
        Loads image-label pairs from a directory structure where each
        subdirectory is a class.
        """
        class_names = sorted([d.name for d in os.scandir(dir_path) if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
        
        self.classes = class_names 
        
        for class_name, class_id in self.class_to_idx.items():
            class_dir = os.path.join(dir_path, class_name)
            for file_name in sorted(os.listdir(class_dir)):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_path = os.path.join(class_dir, file_name)
                    self.samples.append((image_path, class_id))
        logger.info(f"Found classes from directory: {self.class_to_idx}")


    def _load_from_json(self, json_path: str, images_dir: str = "oral1"):
        """
        Loads image-label pairs from a COCO-style JSON file.
        This is your original logic, now encapsulated in a method.
        """
        try:
            with open(json_path, "r") as f:
                dataset = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load or parse JSON {json_path}: {e}")
            return
            
        annotations_map = {ann["image_id"]: ann for ann in dataset.get("annotations", [])}
        categories_map = {cat["id"]: cat for cat in dataset.get("categories", [])}
        sorted_categories = sorted(categories_map.values(), key=lambda x: x['id'])
        self.class_to_idx = {cat["name"]: i for i, cat in enumerate(sorted_categories)}
        self.classes = list(self.class_to_idx.keys())
        base_image_dir = os.path.join(os.path.dirname(json_path), images_dir)

        for image_info in dataset.get("images", []):
            annotation = annotations_map.get(image_info["id"])
            if not annotation:
                continue

            category_id = annotation["category_id"]
            category_info = categories_map.get(category_id)
            if not category_info:
                continue

            class_name = category_info["name"]
            class_idx = self.class_to_idx[class_name]
            
            image_path = os.path.join(base_image_dir, image_info["file_name"])
            self.samples.append((image_path, class_idx))
        logger.info(f"Found classes from JSON: {self.class_to_idx}")


    def __len__(self):
        """Returns the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Retrieves, loads, and transforms a single data sample.
        This is now much simpler.
        """
        image_path, category_id = self.samples[idx]
        image_name = os.path.basename(image_path)
        
        # For compatibility, we can use the index as the image_id
        image_id = idx

        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, IOError) as e:
            logger.error(f"Could not load image {image_path}: {e}. Skipping.")
            return None # Return None if image is corrupted or missing

        if self.transform:
            image = self.transform(image)
            
        return image, category_id, image_id, image_name

    @staticmethod
    def collate_fn(batch):
        """
        A custom collate function that filters out None samples from a batch.
        This is crucial for handling errors during image loading gracefully.
        """
        # Filter out all None items, which may have been returned by __getitem__ on error
        batch = [item for item in batch if item is not None]
        if not batch:
            return None # Return None if the whole batch was invalid
        # Use the default collate function on the filtered batch
        return torch.utils.data.dataloader.default_collate(batch)