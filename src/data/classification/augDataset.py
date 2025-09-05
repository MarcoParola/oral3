import torch
import os
import json
import logging
import random
from PIL import Image, UnidentifiedImageError
from typing import Optional, List, Tuple, Any

from src.data.classification.dataset import OralClassificationDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AugOralClassificationDataset(OralClassificationDataset):
    """
    A specialized version of OralClassificationDataset that includes an
    additional set of augmented data from a specified path.
    """
    
    def __init__(self,
                 data_path: str,
                 augmentation_data_path: str,
                 transform: Optional[callable] = None,
                 augmentation_percentage: float = 100.0):

        super().__init__(data_path, transform)
        logger.info(f"The original dataset was initialized with {len(self.samples)} samples.")

        # Add the 'is_original' flag to the original data.
        self.samples = [(path, label, True) for path, label in self.samples]
        
        if not (0.0 <= augmentation_percentage <= 100.0):
            raise ValueError("The augmentation_percentage value must be between 0.0 and 100.0.")
        
        self._load_augmented_data(augmentation_data_path, augmentation_percentage)

        logger.info(f"Initialization complete. Total samples in the dataset: {len(self.samples)}.")

    def _load_augmented_data(self, aug_path: str, percentage: float):
        """Loads and merges the augmented data."""
        logger.info(f"Starting to load augmented data from: {aug_path}")
        if not os.path.exists(aug_path):
            logger.warning(f"Augmentation data path does not exist: {aug_path}. No data will be added.")
            return

        augmented_samples = self._collect_samples_from_path(aug_path)
        
        if not augmented_samples:
            logger.warning("No valid samples were found at the augmentation path.")
            return

        # Shuffle samples to ensure a random selection.
        random.shuffle(augmented_samples)
        
        num_to_keep = int(len(augmented_samples) * (percentage / 100.0))
        final_augmented_samples = augmented_samples[:num_to_keep]
        
        # Extend the main samples list with the new data.
        self.samples.extend(final_augmented_samples)
        logger.info(f"Loaded {len(final_augmented_samples)} augmented samples ({percentage}%) "
                    f"from a total of {len(augmented_samples)} available.")

        random.shuffle(self.samples)  # Shuffle the combined dataset
        logger.info("Shuffled the combined dataset samples.")


    def _collect_samples_from_path(self, path: str) -> List[Tuple[str, int, bool]]:
        """Collects samples by detecting the source type (directory or JSON)."""
        if os.path.isdir(path):
            return self._collect_from_directory(path)
        elif path.endswith('.json'):
            return self._collect_from_json(path)
        else:
            logger.warning(f"Unrecognized data source type for augmentation: {path}.")
            return []

    def _collect_from_directory(self, dir_path: str) -> List[Tuple[str, int, bool]]:
        """Collects samples from a directory structure."""
        collected = []
        for class_name in os.listdir(dir_path):
            if not os.path.isdir(os.path.join(dir_path, class_name)):
                continue

            # Use the class mapping from the original dataset for consistency.
            if class_name not in self.class_to_idx:
                logger.warning(f"Class '{class_name}' from augmented data was not found "
                               f"in the original dataset and will be skipped.")
                continue
            
            class_id = self.class_to_idx[class_name]
            class_dir = os.path.join(dir_path, class_name)
            for file_name in sorted(os.listdir(class_dir)):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_path = os.path.join(class_dir, file_name)
                    collected.append((image_path, class_id, False)) # False for synthetic data
        return collected

    def _collect_from_json(self, json_path: str, images_dir: str = "oral1") -> List[Tuple[str, int, bool]]:
        """Collects samples from a COCO-style JSON file."""
        collected = []
        try:
            with open(json_path, "r") as f:
                dataset = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load augmentation JSON {json_path}: {e}")
            return []

        annotations_map = {ann["image_id"]: ann for ann in dataset.get("annotations", [])}
        categories_map = {cat["id"]: cat["name"] for cat in dataset.get("categories", [])}
        base_image_dir = os.path.join(os.path.dirname(json_path), images_dir)

        for image_info in dataset.get("images", []):
            annotation = annotations_map.get(image_info["id"])
            if not annotation:
                continue

            category_name = categories_map.get(annotation["category_id"])
            if not category_name or category_name not in self.class_to_idx:
                logger.warning(f"Class '{category_name}' from augmentation JSON not found "
                               f"in the original dataset. Skipping image.")
                continue

            class_idx = self.class_to_idx[category_name]
            image_path = os.path.join(base_image_dir, image_info["file_name"])
            collected.append((image_path, class_idx, False)) # False for synthetic data
        return collected

    def __getitem__(self, idx: int) -> Optional[Tuple[Any, int, int, str, bool]]:
        """Retrieves a single sample from the dataset."""
        image_path, category_id, is_original = self.samples[idx]
        image_name = os.path.basename(image_path)
        
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, IOError) as e:
            logger.error(f"Could not load image {image_path}: {e}. Skipping sample.")
            return None

        if self.transform:
            image = self.transform(image)
        
        return image, category_id, idx, image_name, is_original