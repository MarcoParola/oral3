import hydra
import torch
import pytorch_lightning as pl
import omegaconf
from pathlib import Path

# Import utility functions for transformations and directory management
from src.utils import get_transformations, get_experiment_dirs
# Import logging utilities
from src.log import get_loggers

# Import custom models for various approaches (CAE, DINO, and standard/contrastive classifiers)
from src.models.cae import Autoencoder
from src.models.dino import OralDinoModule
from src.models.classification import OralClassifierModule
from src.models.contrastive_classification import OralContrastiveClassifierModule

# Import data modules corresponding to each model type
from src.data.autoencoder.datamodule import OralAutoencoderDataModule
from src.data.dino.datamodule import OralDinoDataModule
from src.data.classification.datamodule import OralClassificationDataModule
from src.data.contrastive_classification.datamodule import OralContrastiveDataModule

def predict(trainer, model, data, saliency_map_flag, task, classification_mode):
    """
    Executes the testing loop and, if specified, generates saliency maps.
    """
    # Evaluate the model on the test set for classification tasks.
    if task == 'c' or task == 'classification':
        trainer.test(model, datamodule=data)

        # Generate saliency maps if the corresponding flag is set.
        if saliency_map_flag == "grad-cam":
            print("--- Generating Saliency Maps ---")
            predictions = trainer.predict(model, datamodule=data)
            predictions = torch.cat(predictions, dim=0)
            predictions = torch.argmax(predictions, dim=1)
            print("Saliency map generation complete.")


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: omegaconf.DictConfig):
    """
    This main function loads a saved model checkpoint and performs testing on it.
    
    Args:
        cfg (omegaconf.DictConfig): The Hydra configuration object for the current test run.
    """

    # Configure output directories and loggers for the current run.
    run_output_dir, _ = get_experiment_dirs(cfg)
    loggers = get_loggers(cfg, str(run_output_dir))
    
    # Define and validate the base directory containing the saved experiment.
    base_dir = Path(cfg.checkpoint_base_dir)
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint base directory not found: {base_dir}")

    # Construct the path to the original training configuration and verify its existence.
    train_cfg_path = base_dir / ".hydra" / "config.yaml"
    if not train_cfg_path.is_file():
        raise FileNotFoundError(f"Original training config not found: {train_cfg_path}")

    # Load the original training configuration from the specified path.
    print(f"\nLoading original training configuration from: {train_cfg_path}")
    train_cfg = omegaconf.OmegaConf.load(train_cfg_path)

    ckpt_dir = base_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoints directory not found: {ckpt_dir}")
    
    # Determine the specific checkpoint file to be loaded.
    if cfg.checkpoint_file_name:
        # Use the user-specified checkpoint file.
        ckpt_path = ckpt_dir / cfg.checkpoint_file_name
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Specified checkpoint file not found: {ckpt_path}")
    else:
        # Automatically select the most recent checkpoint file if none is specified.
        try:
            ckpt_path = max(ckpt_dir.glob("*.ckpt"), key=lambda f: f.stat().st_mtime)
            print(f"No specific checkpoint file name provided. Automatically using the latest checkpoint found: {ckpt_path.name}")
        except ValueError:
            raise FileNotFoundError(f"No checkpoint files (.ckpt) found in: {ckpt_dir}")

    print(f"\nUsing checkpoint file: {ckpt_path}")

    # Instantiate the model and data module based on the loaded training configuration.
    print(f"\n--- Instantiating Model and DataModule from loaded configuration ---")
    print(f"Original Task: {train_cfg.task}, Original Mode: {train_cfg.classification_mode}")
    
    # Configure dataset paths.
    dataset_dir = Path(train_cfg.dataset.base_path)
    dataset_type = train_cfg.dataset.type
    path_suffix = '.json' if dataset_type == 'json' else ''
    train_data_path = str(dataset_dir / f'train{path_suffix}')
    val_data_path = str(dataset_dir / f'val{path_suffix}')
    test_data_path = str(dataset_dir / f'test{path_suffix}')

    # Retrieve image transformations from the training configuration.
    train_tfms, val_tfms, test_tfms, general_tfms = get_transformations(train_cfg)

    # Map classification modes to their respective model and data module classes.
    MODEL_MAP = {
        'whole': OralClassifierModule,
        'contrastive': OralContrastiveClassifierModule,
        'cae': Autoencoder,
        'dino': OralDinoModule,
    }
    DATA_MODULE_MAP = {
        'whole': OralClassificationDataModule,
        'contrastive': OralContrastiveDataModule,
        'cae': OralAutoencoderDataModule,
        'dino': OralDinoDataModule,
    }
    
    classification_mode = train_cfg.classification_mode
    model_class = MODEL_MAP.get(classification_mode)
    data_module_class = DATA_MODULE_MAP.get(classification_mode)

    if not model_class or not data_module_class:
        raise ValueError(f"Unsupported 'classification_mode' in loaded config: {classification_mode}")


    # Load the model architecture and weights from the specified checkpoint file.
    print(f"Loading model class '{model_class.__name__}' from checkpoint...")
    model = model_class.load_from_checkpoint(ckpt_path)


    # Instantiate the corresponding data module with parameters from the training configuration.
    data_args = {
        'train_path': train_data_path, 
        'val_path': val_data_path, 
        'test_path': test_data_path,
        'batch_size': train_cfg.train.batch_size, 
        'num_workers': train_cfg.train.num_workers,
        'train_transform': train_tfms, 
        'val_transform': val_tfms,
        'test_transform': test_tfms, 
        'transform': general_tfms
    }
    data = data_module_class(**data_args)

    # Setup the data module for testing, with the class labels
    data.setup('test')
    model.classes = data.classes

    trainer = pl.Trainer(logger=loggers, accelerator='auto', devices=train_cfg.train.devices)

    # Execute the prediction and testing loop.
    saliency_method = cfg.get('generate_map')
    print("\n--- Starting Model Testing ---")
    predict(trainer, model, data, saliency_method, train_cfg.task, classification_mode)
    print("\n--- Testing process completed ---")

if __name__ == "__main__":
    main()