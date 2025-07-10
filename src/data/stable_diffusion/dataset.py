# src/data/stable_diffusion/dataset.py
"""
Implements a unified PyTorch Dataset for loading data for diffusion model fine-tuning.
It transparently supports both directory-based and COCO-like JSON data sources.
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

class DiffusionDataset(Dataset):
    """
    A unified PyTorch Dataset for loading image-prompt pairs.

    This dataset can handle:
    - Main data from directories or COCO-style JSON files.
    - Optional regularization data from a directory.
    - Subsampling of the dataset.
    - Dynamic prompt generation with optional randomization to improve model robustness.
    """
    def __init__(self,
                 data_path: str,
                 transform: Optional[v2.Compose] = None,
                 reg_data_path: Optional[str] = None,
                 reg_prompt: str = "a high quality photo of oral cavity",
                 dataset_load_percent: float = 100.0,
                 coco_image_subdir: str = "images",
                 prompt_variation: bool = False
                 ) -> None:

        super().__init__()
        self.transform = transform if transform else self._get_default_transform()
        self.prompt_variation = prompt_variation

        self.prompt_templates = {
            'aphthous': [
                "Medical photograph of oral aphthous ulcer.",
                "Clinical image of aphthous stomatitis.",
                "In-mouth shot of an aphthous lesion."
            ],
            'traumatic': [
                "Medical photograph of oral traumatic lesion.",
                "Clinical image of oral traumatic injury.",
                "In-mouth shot of an oral trauma."
            ],
            'neoplastic': [         
                "Medical photograph of oral neoplastic lesion.",
                "Clinical image of oral neoplastic growth.",
                "In-mouth shot of an oral neoplasm."
            ],
            'cancer': [
                "Medical photograph of oral cancer.",
                "Clinical image of oral carcinoma.",
                "In-mouth shot of oral cancer."
            ],
            'non cancer': [ 
                "Medical photograph of healthy oral tissue.",
                "Clinical image of non-cancerous oral mucosa.",
                "In-mouth shot of healthy mouth."
            ]
        }
        
        self.class_names: List[str] = []
        self.class_to_items_map: Dict[str, List[Tuple[str, str, int]]] = {} 

        logger.info(f"Loading main data from source: {data_path}")
        # Changed main_items_raw to main_items
        main_items = self._load_data(path=data_path, 
                                     coco_image_subdir=coco_image_subdir)
        
        reg_items = []
        if reg_data_path:
            logger.info(f"Loading regularization data from source: {reg_data_path}")
            # Changed reg_items_raw to reg_items
            reg_items = self._load_data(path=reg_data_path,
                                        directory_prompt_override=reg_prompt,
                                        coco_image_subdir=coco_image_subdir)

        # Apply subsampling
        main_items_subsampled = self._subsample_items(main_items, dataset_load_percent, "main")
        reg_items_subsampled = self._subsample_items(reg_items, dataset_load_percent, "regularization")

        self.all_items = main_items_subsampled + reg_items_subsampled
        
        self._build_class_maps()
        
        self.prompt_list = [self._generate_prompt_for_class(cls_name, use_variation=False) for cls_name in self.class_names]
        if self.prompt_list:
            logger.debug(f"Discovered Prompts (one default per class): {self.prompt_list}")

        logger.info(f"Dataset initialized. Total items: {len(self.all_items)} "
                    f"({len(main_items_subsampled)} main, {len(reg_items_subsampled)} regularization).")

    def _build_class_maps(self):
        """
        Builds class_names and class_to_items_map from self.all_items.
        Assumes each item in `self.all_items` is a tuple `(image_path, prompt, class_name_str)`.
        """
        self.class_names.clear()
        self.class_to_items_map.clear()
        unique_class_names_set = set()

        for current_idx, (image_path, text_prompt, class_name_str) in enumerate(self.all_items):
            unique_class_names_set.add(class_name_str)
            if class_name_str not in self.class_to_items_map:
                self.class_to_items_map[class_name_str] = []
            
            self.class_to_items_map[class_name_str].append((image_path, text_prompt, current_idx))

        self.class_names = sorted(list(unique_class_names_set))
        logger.info(f"Discovered {len(self.class_names)} unique classes: {self.class_names}")


    def _generate_prompt_for_class(self, class_name: str, use_variation: Optional[bool] = None) -> str:
        """
        Generates a descriptive prompt for a given class name, with optional random variation.
        `use_variation` overrides `self.prompt_variation` if specified.
        """
        class_name_lower = class_name.lower().strip().replace('_', ' ')
        
        default_prompt = f"high quality picture of oral cavity with {class_name_lower}"
        template_list = self.prompt_templates.get(class_name_lower, [default_prompt])
        
        if not template_list: 
             logger.warning(f"Empty template list for class '{class_name}'. Using a generic prompt.")
             return default_prompt

        should_vary = self.prompt_variation if use_variation is None else use_variation

        if not should_vary:
            return template_list[0]
        else:
            return random.choice(template_list)


    def _load_data(self, path: str, coco_image_subdir: str, directory_prompt_override: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """Dispatches data loading based on the source path type."""
        if not os.path.exists(path):
            logger.error(f"Data source path not found: {path}")
            return []
            
        if os.path.isdir(path):
            logger.info(f"Path is a directory. Loading with directory-based strategy.")
            return self._load_from_directory(path, prompt_override=directory_prompt_override)
        elif path.endswith('.json'):
            logger.info(f"Path is a JSON file. Loading with COCO-style strategy.")
            return self._load_from_coco(path, coco_image_subdir)
        else:
            logger.warning(f"Unrecognized data source type: {path}. Cannot load data.")
            return []

    def _load_from_directory(self, dir_path: str, prompt_override: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """
        Loads data from a directory structure where subdirectories are class names.
        Each item returned includes (image_path, prompt, actual_class_name).
        """
        items = []
        for class_name_dir in sorted(os.listdir(dir_path)):
            class_dir = os.path.join(dir_path, class_name_dir)
            if not os.path.isdir(class_dir):
                continue
            
            actual_class_name = class_name_dir

            prompt = prompt_override if prompt_override else self._generate_prompt_for_class(actual_class_name, use_variation=False)

            for file_name in sorted(os.listdir(class_dir)):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    image_path = os.path.join(class_dir, file_name)
                    items.append((image_path, prompt, actual_class_name))
        
        if not items:
            logger.warning(f"No image files found in directory: {dir_path}")
        return items

    def _load_from_coco(self, json_path: str, image_subdir: str) -> List[Tuple[str, str, str]]:
        """
        Loads data from a COCO-style JSON annotation file.
        Each item returned includes (image_path, prompt, category_name).
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load or parse JSON file {json_path}: {e}")
            return []

        if not all(k in manifest for k in ["images", "annotations", "categories"]):
            logger.error("COCO JSON is malformed. Missing 'images', 'annotations', or 'categories' keys.")
            return []

        base_image_dir = os.path.join(os.path.dirname(json_path), image_subdir)
        annotations_map = {ann["image_id"]: ann for ann in manifest.get("annotations", [])}
        category_map = {cat["id"]: cat["name"] for cat in manifest.get("categories", [])}
        
        items = []
        for image_info in manifest.get("images", []):
            image_id = image_info.get("id")
            file_name = image_info.get("file_name")
            if not (image_id and file_name):
                continue

            annotation = annotations_map.get(image_id)
            if not annotation:
                continue

            category_id = annotation.get("category_id")
            category_name = category_map.get(category_id)
            if not category_name:
                continue
            
            image_path = os.path.join(base_image_dir, file_name)
            prompt = self._generate_prompt_for_class(category_name, use_variation=False)
            items.append((image_path, prompt, category_name))
        
        if not items:
            logger.warning(f"No valid image-annotation pairs were found in JSON file: {json_path}")
        return items

    def _subsample_items(self, items: list, percent: float, name: str) -> list:
        """Filters a list of items to a given percentage."""
        if not (0.0 <= percent <= 100.0):
            percent = max(0.0, min(100.0, percent))
            logger.warning(f"dataset_load_percent was clamped to {percent}%")
        
        if percent >= 100.0:
            return items
            
        num_to_keep = int(len(items) * (percent / 100.0))
        logger.info(f"Applying {percent}% load to {name} data: retaining {num_to_keep} of {len(items)} items.")
        return items[:num_to_keep]

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.all_items)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves, loads, and v2 a single data sample.
        """
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self)}")

        image_path, text_prompt, _ = self.all_items[idx] 
        
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, IOError) as e:
            logger.error(f"Failed to load image {image_path} at index {idx}: {e}. Skipping sample.")
            return None

        pixel_values = self.transform(image)

        return {
            "pixel_values": pixel_values,
            "text": text_prompt,
            "image_path": image_path,
        }

    def get_class_names(self) -> List[str]:
        """Returns a list of unique class names found in the dataset."""
        return self.class_names

    def get_items_for_class(self, class_name: str) -> List[Tuple[str, str, int]]:
        """
        Returns a list of (image_path, prompt, original_index_in_all_items) tuples for a given class.
        """
        return self.class_to_items_map.get(class_name, [])
    
    def get_prompts_for_class(self, class_name: str, count: int) -> List[str]:
        """
        Generates `count` prompts for a given class, using the _generate_prompt_for_class logic.
        This allows generating multiple (potentially varied) prompts per class.
        """
        if class_name not in self.class_names:
            logger.warning(f"Class '{class_name}' not found in dataset. Cannot generate prompts.")
            return []
        
        generated_prompts = []
        for _ in range(count):
            generated_prompts.append(self._generate_prompt_for_class(class_name, use_variation=True))
        return generated_prompts

    def get_random_images_for_class(self, class_name: str, count: int, random_seed: Optional[int] = None) -> List[torch.Tensor]:
        """
        Randomly samples and loads `count` images for a given class.
        These images are loaded on the fly, which might be slow for many samples.
        """
        class_items = self.class_to_items_map.get(class_name, [])
        if not class_items:
            logger.warning(f"No items found for class '{class_name}'. Cannot sample images.")
            return []

        num_available = len(class_items)
        actual_count = min(count, num_available)
        
        if actual_count == 0:
            return []

        if random_seed is not None:
            temp_random_state = random.getstate()
            random.seed(random_seed)
        
        sampled_items = random.sample(class_items, actual_count)
        
        if random_seed is not None:
            random.setstate(temp_random_state)

        loaded_images = []
        for image_path, _, _ in tqdm(sampled_items, desc=f"Loading random {class_name} images"):
            try:
                image = Image.open(image_path).convert("RGB")
                pixel_values = self.transform(image)
                loaded_images.append(pixel_values)
            except (FileNotFoundError, UnidentifiedImageError, IOError) as e:
                logger.error(f"Failed to load sampled image {image_path} for class {class_name}: {e}. Skipping.")
        
        return loaded_images

    @staticmethod
    def collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        """
        A custom collate function that filters out None samples before batching.
        """
        valid_batch = [item for item in batch if item is not None]
        if not valid_batch:
            return None
        return torch.utils.data.dataloader.default_collate(valid_batch)

    def _get_default_transform(self) -> v2.Compose:
        """Provides a default transformation pipeline if none is specified."""
        logger.warning("No transform provided")
        return v2.Compose([
            v2.Resize((256, 256), interpolation=v2.InterpolationMode.BICUBIC),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])