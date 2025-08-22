import logging
from typing import Optional

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from src.data.classification.dataset import OralClassificationDataset
from src.data.classification.augDataset import AugOralClassificationDataset

log = logging.getLogger(__name__)

class OralClassificationDataModule(LightningDataModule):
    """
    This DataModule supports two modes for the training set: a standard mode
    using a single data source, and an augmented mode that merges the primary
    dataset with an additional collection of augmented data. The operational
    mode is determined by the presence of the `train_aug_path` parameter.
    """

    def __init__(self,
                 train_path: str,
                 val_path: str,
                 test_path: str,
                 train_aug_path: Optional[str] = None,
                 train_aug_percentage: Optional[float] = 100.0,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 train_transform: Optional[callable] = None,
                 val_transform: Optional[callable] = None,
                 test_transform: Optional[callable] = None,
                 transform: Optional[callable] = None):
        """
        Initializes the DataModule.

        Args:
            train_path (str): Path to the primary training data source.
            val_path (str): Path to the validation data source.
            test_path (str): Path to the test data source.
            train_aug_path (str, optional): Path to the augmented training data.
            train_aug_percentage (float): Percentage of the augmented dataset to use.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses for data loading.
            train_transform (callable, optional): Transformations for the training set.
            val_transform (callable, optional): Transformations for the validation set.
            test_transform (callable, optional): Transformations for the test set.
            transform (callable, optional): Default transformation for all sets.
        """
        super().__init__()

        # Assign data paths and augmentation parameters.
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.train_aug_path = train_aug_path
        self.train_aug_percentage = train_aug_percentage
        
        # Assign dataloader parameters.
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Assign transforms, using the generic 'transform' as a fallback.
        self.train_transform = train_transform if train_transform is not None else transform
        self.val_transform = val_transform if val_transform is not None else transform
        self.test_transform = test_transform if test_transform is not None else transform

        # Initialize attributes for class information and prediction dataset.
        self.classes = None
        self.predict_dataset = None 

    @property
    def num_classes(self) -> int:
        """Returns the number of classes in the dataset."""
        return len(self.classes) if self.classes is not None else 0

    def setup(self, stage: str = None):
        """
        Instantiates the required datasets based on the execution stage ('fit',
        'test', or 'predict'), as called by the PyTorch Lightning Trainer.

        Args:
            stage (str, optional): The current stage of execution.
        """
        
        # Setup for training and validation stages.
        if stage == 'fit' or stage is None:
            if self.train_aug_path:
                log.info(f"Augmentation path provided. Initializing 'AugOralClassificationDataset' for training.")
                self.train_dataset = AugOralClassificationDataset(
                    data_path=self.train_path,
                    augmentation_data_path=self.train_aug_path,
                    augmentation_percentage=self.train_aug_percentage,
                    transform=self.train_transform
                )
            else:
                log.info(f"No augmentation path provided. Initializing 'OralClassificationDataset' for training.")
                self.train_dataset = OralClassificationDataset(
                    self.train_path,
                    transform=self.train_transform
                )

            self.val_dataset = OralClassificationDataset(self.val_path, transform=self.val_transform)
            self.classes = self.train_dataset.classes

        # Setup for the testing stage.
        if stage == 'test' or stage is None:
            self.test_dataset = OralClassificationDataset(self.test_path, transform=self.test_transform)
            if self.classes is None:
                self.classes = self.test_dataset.classes
        
        # Setup for the prediction stage.
        if stage == 'predict':
            if not hasattr(self, 'test_dataset'): self.setup('test')
            self.predict_dataset = self.test_dataset

    def _create_dataloader(self, dataset, shuffle: bool = False) -> DataLoader:
        """
        A private helper method to create a DataLoader with consistent settings.

        Args:
            dataset (Dataset): The dataset from which to load the data.
            shuffle (bool): Whether to shuffle the data at every epoch.

        Returns:
            DataLoader: The configured DataLoader instance.
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=OralClassificationDataset.collate_fn
        )

    def train_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the training set."""
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the validation set."""
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the test set."""
        return self._create_dataloader(self.test_dataset, shuffle=False)
    
    def predict_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the prediction set."""
        return self._create_dataloader(self.predict_dataset, shuffle=False)