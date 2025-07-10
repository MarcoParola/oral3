# In: src/data/classification/datamodule.py

import hydra
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from src.data.classification.dataset import OralClassificationDataset


class OralClassificationDataModule(LightningDataModule):
    def __init__(self, train, val, test, batch_size=32, num_workers=0,
                 train_transform=None, val_transform=None, test_transform=None, transform=None):
        super().__init__()
        # This part handles assigning the correct transform if only a base one is provided
        self.train_transform = train_transform if train_transform is not None else transform
        self.val_transform = val_transform if val_transform is not None else transform
        self.test_transform = test_transform if test_transform is not None else transform

        # Store paths and parameters
        self.train_path = train
        self.val_path = val
        self.test_path = test
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.classes = None  
        
    @property
    def num_classes(self):
        """Derives the number of classes from the length of the classes list."""
        return len(self.classes) if self.classes is not None else 0
        
    def setup(self, stage: str = None):
        """Instantiate datasets and capture class information from them."""
        
        # Setup for training and validation
        if stage == 'fit' or stage is None:
            self.train_dataset = OralClassificationDataset(self.train_path, transform=self.train_transform)
            self.val_dataset = OralClassificationDataset(self.val_path, transform=self.val_transform)
            self.classes = self.train_dataset.classes

        # Setup for testing
        if stage == 'test' or stage is None:
            self.test_dataset = OralClassificationDataset(self.test_path, transform=self.test_transform)
            if self.classes is None:
                self.classes = self.test_dataset.classes

        # Setup for prediction
        if stage == 'predict':
            # Assume predict uses the test_dataset
            if not hasattr(self, 'test_dataset'): self.setup('test')
            self.predict_dataset = self.test_dataset


    def _create_dataloader(self, dataset, shuffle=False):
        """Helper function to create a DataLoader."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=OralClassificationDataset.collate_fn # Use the custom collate_fn
        )

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self):
        return self._create_dataloader(self.predict_dataset, shuffle=False)