import hydra
import torch
import pytorch_lightning as pl
import omegaconf
from pathlib import Path

from src.utils import *
from src.log import get_loggers
from src.utils import get_experiment_dirs

from src.models.classification import OralClassifierModule
from src.data.classification.datamodule import OralClassificationDataModule
from src.models.contrastive_classification import OralContrastiveClassifierModule
from src.data.contrastive_classification.datamodule import OralContrastiveDataModule
from src.models.cae import Autoencoder
from src.data.autoencoder.datamodule import OralAutoencoderDataModule
from src.models.dino import OralDinoModule
from src.data.dino.datamodule import OralDinoDataModule

def predict(trainer, model, data, saliency_map_flag, task, classification_mode):
    if task == 'c' or task == 'classification':      
        trainer.test(model, datamodule=data)
        if saliency_map_flag == "grad-cam":
            print("--- Generating Saliency Maps ---")
            predictions = trainer.predict(model, datamodule=data)
            predictions = torch.cat(predictions, dim=0)
            predictions = torch.argmax(predictions, dim=1)
            print("Saliency map generation complete.")


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):
    """
    This main function loads a saved model checkpoint and performs testing on it.
    
    Args:
        cfg (omegaconf.DictConfig): The Hydra configuration object for the current test run.
    """
    print("--- Starting Checkpoint Loading and Testing Process ---")

    # Directories for experiment outputs 
    run_output_dir, _ = get_experiment_dirs(cfg)

    # Setup loggers
    loggers = get_loggers(cfg, str(run_output_dir))

    base_dir_str = cfg.get('checkpoint_base_dir')
    ckpt_file_name = cfg.get('checkpoint_file_name')


    if base_dir_str is None:
        print("Error: 'checkpoint_base_dir' is not provided in the configuration.")
        raise ValueError("'checkpoint_base_dir' is required; it should point to the root output directory of a training run.")


    base_dir = Path(base_dir_str)
    
    if not base_dir.is_dir():
        print(f"Error: Checkpoint base directory does not exist or is not a directory: {base_dir}")
        raise FileNotFoundError(f"Checkpoint base directory not found: {base_dir}")


    train_cfg_path = base_dir / ".hydra" / "config.yaml"


    if not train_cfg_path.is_file():
        print(f"Error: Original training configuration file not found at: {train_cfg_path}")
        print("Please ensure 'checkpoint_base_dir' points to the correct training run output directory.")
        raise FileNotFoundError(f"Original training config not found: {train_cfg_path}")


    print(f"\nLoading original training configuration from: {train_cfg_path}")
    train_cfg = omegaconf.OmegaConf.load(train_cfg_path)


    ckpt_dir = base_dir / "checkpoints"
    
    if not ckpt_dir.is_dir():
        print(f"Error: Checkpoints directory not found within the base directory: {ckpt_dir}")
        raise FileNotFoundError(f"Checkpoints directory not found: {ckpt_dir}")


    ckpt_path = None
    if ckpt_file_name is not None:
        ckpt_path = ckpt_dir / ckpt_file_name
        if not ckpt_path.is_file():
            print(f"Error: The specified checkpoint file was not found: {ckpt_path}")
            raise FileNotFoundError(f"Specific checkpoint file not found: {ckpt_path}")
    else:
        try:
            # Find the latest checkpoint by modification time
            ckpt_path = max(ckpt_dir.glob("*.ckpt"), key=lambda f: f.stat().st_mtime)
            print(f"No specific checkpoint file name provided. Automatically using the latest checkpoint found: {ckpt_path}")
        except ValueError: # No .ckpt files found by max()
            print(f"Error: No checkpoint files (.ckpt) found in: {ckpt_dir}")
            raise FileNotFoundError(f"No checkpoint files found in: {ckpt_dir}")

    print(f"\nUsing checkpoint file: {ckpt_path}")

    trainer = pl.Trainer(
        logger=loggers,
        accelerator='auto',
        devices=train_cfg.train.devices
    )

    model = None
    data = None
    
    print(f"\n--- Instantiating Model and DataModule from loaded configuration ---")
    print(f"Original Task: {train_cfg.task}, Original Mode: {train_cfg.classification_mode}")

    dataset_path = Path(train_cfg.dataset.base_path) / train_cfg.dataset.dataset_name / train_cfg.dataset.augmentation_type

    if train_cfg.dataset.data_format == 'json':
        train_data_path = str(dataset_path / 'train.json')
        val_data_path = str(dataset_path / 'val.json')
        test_data_path = str(dataset_path / 'test.json')
    elif train_cfg.dataset.data_format == 'directory':
        train_data_path = str(dataset_path / 'train')
        val_data_path = str(dataset_path / 'val')
        test_data_path = str(dataset_path / 'test')
    else:
        raise ValueError(f"Unsupported 'data_format' in the loaded training config: {train_cfg.dataset.data_format}")
    

    train_tfms, val_tfms, test_tfms, general_tfms = get_transformations(train_cfg)
    

    if train_cfg.task == 'c' or train_cfg.task == 'classification':
        
        if train_cfg.classification_mode == 'whole':
            model = OralClassifierModule.load_from_checkpoint(ckpt_path)
            data = OralClassificationDataModule(
                train=train_data_path, val=val_data_path, test=test_data_path,
                batch_size=train_cfg.train.batch_size, num_workers=train_cfg.train.num_workers,
                train_transform=train_tfms, val_transform=val_tfms,
                test_transform=test_tfms, transform=general_tfms
            )
        
        elif train_cfg.classification_mode == 'contrastive':
            model = OralContrastiveClassifierModule.load_from_checkpoint(ckpt_path)
            data = OralContrastiveDataModule(
                train=train_data_path, val=val_data_path, test=test_data_path,
                batch_size=train_cfg.train.batch_size, num_workers=train_cfg.train.num_workers,
                train_transform=train_tfms, val_transform=val_tfms,
                test_transform=test_tfms, transform=general_tfms
            )

        elif train_cfg.classification_mode == 'cae':
            model = Autoencoder.load_from_checkpoint(ckpt_path)
            data = OralAutoencoderDataModule(
                train=train_data_path, val=val_data_path, test=test_data_path,
                batch_size=train_cfg.train.batch_size, num_workers=train_cfg.train.num_workers,
                train_transform=train_tfms, val_transform=val_tfms,
                test_transform=test_tfms, transform=general_tfms
            )

        elif train_cfg.classification_mode == 'dino':
            model = OralDinoModule.load_from_checkpoint(ckpt_path)
            data = OralDinoDataModule(
                train=train_data_path, val=val_data_path, test=test_data_path,
                batch_size=train_cfg.train.batch_size, num_workers=train_cfg.train.num_workers,
                train_transform=train_tfms, val_transform=val_tfms,
                test_transform=test_tfms, transform=general_tfms
            )
        else:
            raise ValueError(f"Unsupported 'classification_mode' in loaded config: {train_cfg.classification_mode}")
    else:
        raise ValueError(f"Unsupported 'task' in loaded config: {train_cfg.task}. Only 'c' or 'classification' are currently supported.")

    
    if model is not None and data is not None:
        saliency_method = train_cfg.get('generate_map') 
        
        print("\n--- Starting Model Prediction / Testing ---")
        predict(trainer, model, data, saliency_method, train_cfg.task, train_cfg.classification_mode)
        print("\n--- Prediction / Testing process completed ---")
    else:
        print(f"Error: Failed to load model or data for mode: {train_cfg.classification_mode}. Aborting test.")
        return
    

if __name__ == "__main__":
    main()