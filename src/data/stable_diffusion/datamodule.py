# src/data/stable_diffusion/datamodule.py
"""
Implements the PyTorch Lightning DataModule for the UnifiedDiffusionDataset.
This module encapsulates all data-related setup, including dataset and
dataloader instantiation, and manages data transformations via configuration.
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import logging
from typing import Optional, Union, Tuple

from torchvision import transforms
from omegaconf import DictConfig, ListConfig, OmegaConf 
import hydra 

from src.data.stable_diffusion.dataset import DiffusionDataset

logger = logging.getLogger(__name__)

class DiffusionDataModule(pl.LightningDataModule):
    """A PyTorch Lightning DataModule for the UnifiedDiffusionDataset."""
    def __init__(self,
                 train_path: str,
                 val_path: str,
                 test_path: str,
                 reg_data_path: Optional[str] = None,
                 reg_prompt: str = "a high quality photo of oral cavity",
                 coco_image_subdir: str = "images",
                 dataset_load_percent: float = 100.0,
                 batch_size: int = 16,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_transform: Optional[Union[ListConfig, DictConfig]] = None,
                 val_transform: Optional[Union[ListConfig, DictConfig]] = None,
                 test_transform: Optional[Union[ListConfig, DictConfig]] = None,
                 base_transform: Optional[Union[ListConfig, DictConfig]] = None,
                ) -> None:
        """Initializes the DataModule."""
        super().__init__()
        self.save_hyperparameters()
        
        logger.info("DataModule is instantiating transforms from configuration.")
        self.train_transform_inst, self.val_transform_inst, self.test_transform_inst = self._init_transforms()
        
        self.train_dataset: Optional[DiffusionDataset] = None
        self.val_dataset: Optional[DiffusionDataset] = None
        self.test_dataset: Optional[DiffusionDataset] = None
        
        if self.hparams.num_workers == 0 and self.hparams.persistent_workers:
            logger.warning("persistent_workers=True requires num_workers > 0. Overriding to False.")
            self.hparams.persistent_workers = False

    def setup(self, stage: Optional[str] = None) -> None:
        """Assigns and prepares datasets for the specified stage."""
        logger.info(f"DataModule setup initiated for stage: {stage}")
        
        dataset_common_args = {"coco_image_subdir": self.hparams.coco_image_subdir}
        
        if stage in ("fit", None):
            self.train_dataset = DiffusionDataset(
                data_path=self.hparams.train_path,
                transform=self.train_transform_inst,
                reg_data_path=self.hparams.reg_data_path,
                reg_prompt=self.hparams.reg_prompt,
                dataset_load_percent=self.hparams.dataset_load_percent,
                **dataset_common_args
            )
            logger.info(f"Training dataset instantiated with {len(self.train_dataset)} samples.")

            self.val_dataset = DiffusionDataset(
                data_path=self.hparams.val_path,
                transform=self.val_transform_inst,
                **dataset_common_args
            )
            logger.info(f"Validation dataset instantiated with {len(self.val_dataset)} samples.")

        if stage in ("test", None):
            self.test_dataset = DiffusionDataset(
                data_path=self.hparams.test_path,
                transform=self.test_transform_inst,
                **dataset_common_args
            )
            logger.info(f"Test dataset instantiated with {len(self.test_dataset)} samples.")

        if stage == "predict":
            if self.test_dataset is None: self.setup("test")
            self.predict_dataset = self.test_dataset
            logger.info(f"Prediction dataset assigned (size: {len(self.predict_dataset) if self.predict_dataset else 0}).")

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.train_dataset, shuffle=True, drop_last=True)
    
    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.val_dataset, shuffle=False)
    
    def test_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.test_dataset, shuffle=False)
    
    def predict_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.predict_dataset, shuffle=False)
    
    def _create_dataloader(self, dataset: Optional[DiffusionDataset], shuffle: bool, drop_last: bool = False) -> DataLoader:
        """A factory method for creating DataLoader instances."""
        if not dataset:
            logger.warning(f"Returning an empty DataLoader for a None or empty dataset.")
            return DataLoader([])
        
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size, shuffle=shuffle,
            num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory,
            persistent_workers=(self.hparams.persistent_workers and self.hparams.num_workers > 0),
            drop_last=drop_last, collate_fn=DiffusionDataset.collate_fn
        )
    
    def _init_transforms(self) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
        """Initializes all transform pipelines from the configuration."""
        base_t = self._instantiate_transforms(self.hparams.base_transform)
        train_t = self._instantiate_transforms(self.hparams.train_transform) or base_t
        val_t = self._instantiate_transforms(self.hparams.val_transform) or base_t
        test_t = self._instantiate_transforms(self.hparams.test_transform) or base_t

        if train_t is None: raise ValueError("Training transform is required but could not be instantiated.")
        if val_t is None: raise ValueError("Validation transform is required but could not be instantiated.")
        if test_t is None: logger.warning("Test/Predict transform is not configured, using base transform.")
        
        return train_t, val_t, test_t

    def _instantiate_transforms(self, cfg: Optional[Union[ListConfig, DictConfig]]) -> Optional[transforms.Compose]:
        """
        Instantiates a torchvision transforms pipeline from a Hydra/OmegaConf config.
        This version intelligently handles string-based interpolation modes.
        """
        if cfg is None: return None
        
        transform_list = []
        if not OmegaConf.is_list(cfg): 
            cfg = [cfg]
        
        interpolation_map = {
            "NEAREST": transforms.InterpolationMode.NEAREST,
            "BILINEAR": transforms.InterpolationMode.BILINEAR,
            "BICUBIC": transforms.InterpolationMode.BICUBIC,
            "LANCZOS": transforms.InterpolationMode.LANCZOS,
        }

        for t_cfg in cfg:
            if t_cfg.get('_target_') == 'torchvision.transforms.Resize' and \
               'interpolation' in t_cfg and isinstance(t_cfg.interpolation, str):
                interp_str = t_cfg.interpolation.upper()
                if interp_str in interpolation_map:
                    logger.info(f"Converting interpolation string '{t_cfg.interpolation}' to enum for Resize transform.")
                    t_cfg.interpolation = interpolation_map[interp_str]
                else:
                    logger.warning(f"Unrecognized interpolation string: '{t_cfg.interpolation}'. Letting torchvision handle it.")
            try:
                instance = hydra.utils.instantiate(t_cfg)
                transform_list.append(instance)
            except Exception as e:
                logger.error(f"Failed to instantiate transform from config: {t_cfg}. Error: {e}")
        return transforms.Compose(transform_list) if transform_list else None