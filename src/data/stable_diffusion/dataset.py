"""
Implements a PyTorch Dataset for loading data for diffusion model fine-tuning.
It transparently supports both directory-based and COCO-like JSON data sources.
"""

# --- Standard Library Imports ---
import os
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
import random

# --- Third-party Imports ---
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image, UnidentifiedImageError

# Configure a module-level logger.
logger = logging.getLogger(__name__)

class DiffusionDataset(Dataset):
    """
    A dataset class designed for fine-tuning diffusion models, which handles
    data loading from multiple formats, prompt generation, and optional inclusion
    of regularization data.
    """
    def __init__(self,
                 data_path: str,
                 transform: Optional[v2.Compose] = None,
                 reg_data_path: Optional[str] = None,
                 reg_prompt: str = "a high quality photo of oral cavity",
                 dataset_load_percent: float = 100.0,
                 image_folder: Optional[str] = "oral1",
                 prompt_variation: bool = False,
                 return_class_label: bool = False
                 ) -> None:
       
        super().__init__()
        self.transform = transform if transform else self._get_default_transform()
        self.prompt_variation = prompt_variation
        self.return_class_label = return_class_label

        self.prompt_templates = {
            'aphthous': [
                "a medical photograph of an aphthous lesion",
                "a clinical image of an aphthous lesion",
                "in-mouth shot of an aphthous lesion",
                "high-resolution close-up of an aphthous ulcer",
                "detailed intraoral photo showing an aphthous lesion"
            ],
            'traumatic': [
                "a medical photograph of a traumatic lesion",
                "a clinical image of a traumatic lesion",
                "in-mouth shot of a traumatic lesion",
                "close-up clinical shot of a traumatic mouth wound",
                "detailed intraoral photo of a traumatic tissue injury"
            ],
            'neoplastic': [
                "a medical photograph of a neoplastic lesion",
                "a clinical image of a neoplastic lesion",
                "in-mouth shot of a neoplastic lesion",
                "biopsy-level clinical photo of a neoplastic oral lesion",
                "high-definition mouth image showing neoplastic growth"
            ],
            'cancer': [
                "a medical photograph of a cancerous tissue",
                "a clinical image of a cancerous tissue",
                "in-mouth shot of a cancerous tissue",
                "detailed intraoral photo of cancerous tissue",
                "high-contrast clinical shot of cancerous lesion"
            ],
            'non cancer': [
                "a medical photograph of healthy tissue",
                "a clinical image of healthy tissue",
                "in-mouth shot of healthy tissue",
                "clear clinical photo of healthy oral mucosa",
                "well-lit intraoral image of healthy mouth tissue"
            ]
        }
                
        # Initialize data structures for class and sample management.
        self.class_names: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.class_to_items_map: Dict[str, List[Tuple[str, str, int]]] = {}

        # Load main and optional regularization datasets.
        main_items = self._load_data(path=data_path, image_folder=image_folder)
        
        reg_items = []
        if reg_data_path:
            reg_items = self._load_data(path=reg_data_path,
                                         directory_prompt_override=reg_prompt,
                                         image_folder=image_folder)
            
        # Subsample datasets according to the specified percentage.
        main_items_subsampled = self._subsample_items(main_items, dataset_load_percent, "main")
        reg_items_subsampled = self._subsample_items(reg_items, dataset_load_percent, "regularization")

        self.all_items = main_items_subsampled + reg_items_subsampled

        # Build mappings from class names to indices and items.
        self._build_class_maps()
        
        self.prompt_list = [self._generate_prompt_for_class(cls_name, use_variation=False) for cls_name in self.class_names]
        if self.prompt_list:
            logger.debug(f"Discovered Prompts (one default per class): {self.prompt_list}")

        logger.info(f"Dataset initialized. Total items: {len(self.all_items)} "
                    f"({len(main_items_subsampled)} main, {len(reg_items_subsampled)} regularization).")

    def _build_class_maps(self):
        """
        Constructs internal mappings for class names, indices, and their
        corresponding data samples.
        """
        self.class_names.clear()
        self.class_to_items_map.clear()
        unique_class_names_set = set()

        for current_idx, (image_path, text_prompt, class_name_str) in enumerate(self.all_items):
            unique_class_names_set.add(class_name_str)
            if class_name_str not in self.class_to_items_map:
                self.class_to_items_map[class_name_str] = []
            
            self.class_to_items_map[class_name_str].append((image_path, text_prompt, current_idx))

        # Create a sorted, reproducible list of class names and a map to integer indices.
        self.class_names = sorted(list(unique_class_names_set))
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        
        logger.info(f"Discovered {len(self.class_names)} unique classes: {self.class_names}")
        logger.info(f"Class to index mapping: {self.class_to_idx}")

    def _generate_prompt_for_class(self, class_name: str, use_variation: Optional[bool] = None) -> str:
        """
        Selects or generates a text prompt for a given class, optionally choosing
        from a template list to introduce variation.
        """
        class_name_lower = class_name.lower().strip().replace('_', ' ')
        default_prompt = f"high quality picture of oral cavity with {class_name_lower}"
        template_list = self.prompt_templates.get(class_name_lower, [default_prompt])
        
        if not template_list: 
             return default_prompt

        should_vary = self.prompt_variation if use_variation is None else use_variation
        return random.choice(template_list) if should_vary else template_list[0]

    def _load_data(self, path: str, image_folder: str, directory_prompt_override: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """Dispatches to the appropriate data loading method based on the source path type."""
        if not os.path.exists(path):
            return []
        if os.path.isdir(path):
            return self._load_from_directory(path, prompt_override=directory_prompt_override)
        elif path.endswith('.json'):
            return self._load_from_coco(path, image_folder)
        return []

    def _load_from_directory(self, dir_path: str, prompt_override: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """Loads data from a directory structure where subdirectories are class names."""
        items = []
        for class_name_dir in sorted(os.listdir(dir_path)):
            class_dir = os.path.join(dir_path, class_name_dir)
            if not os.path.isdir(class_dir):
                continue
            
            actual_class_name = class_name_dir
            prompt = prompt_override if prompt_override else self._generate_prompt_for_class(actual_class_name)

            for file_name in sorted(os.listdir(class_dir)):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    image_path = os.path.join(class_dir, file_name)
                    items.append((image_path, prompt, actual_class_name))
        return items

    def _load_from_coco(self, json_path: str, image_subdir: str) -> List[Tuple[str, str, str]]:
        """Loads data from a COCO-style JSON annotation file."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            return []

        base_image_dir = os.path.join(os.path.dirname(json_path), image_subdir)
        annotations_map = {ann["image_id"]: ann for ann in manifest.get("annotations", [])}
        category_map = {cat["id"]: cat["name"] for cat in manifest.get("categories", [])}
        
        items = []
        for image_info in manifest.get("images", []):
            image_id, file_name = image_info.get("id"), image_info.get("file_name")
            annotation = annotations_map.get(image_id)
            category_id = annotation.get("category_id") if annotation else None
            category_name = category_map.get(category_id) if category_id else None
            if not all([image_id, file_name, annotation, category_id, category_name]):
                continue
            
            image_path = os.path.join(base_image_dir, file_name)
            prompt = self._generate_prompt_for_class(category_name)
            items.append((image_path, prompt, category_name))
        return items

    def _subsample_items(self, items: list, percent: float, name: str) -> list:
        if percent >= 100.0: return items
        percent = max(0.0, min(100.0, percent))
        num_to_keep = int(len(items) * (percent / 100.0))
        return items[:num_to_keep]

    def __len__(self) -> int:
        return len(self.all_items)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")

        image_path, prompt, class_name = self.all_items[idx]
        
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, IOError) as e:
            logger.error(f"Failed to load image {image_path}: {e}. Skipping.")
            return None

        pixel_values = self.transform(image)

        item = {
            "images": pixel_values,
            "prompts": prompt,
            "image_path": image_path,
        }

        if self.return_class_label:
            item["class_labels"] = self.class_to_idx[class_name]
        
        return item

    @staticmethod
    def collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        valid_batch = [item for item in batch if item is not None]
        if not valid_batch:
            return None
        return torch.utils.data.dataloader.default_collate(valid_batch)

    def _get_default_transform(self) -> v2.Compose:
        return v2.Compose([
            v2.Resize((256, 256), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])