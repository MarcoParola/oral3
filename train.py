import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb 

# Import utility functions
import os 
from datetime import datetime as dt
from src.utils import get_early_stopping, get_experiment_dirs, get_transformations
from src.log import LossLogCallback, get_loggers
from omegaconf import DictConfig
from typing import Tuple, Dict

# Import custom models for various approaches (CAE, MAE, DINO, VICReg, MOCO) from the project
from src.models.cae import Autoencoder
from src.models.mae import OralMAEModule
from src.models.dino import OralDinoModule
from src.models.vicreg import OralVICRegModule
from src.models.moco import OralMOCOModule

# Import data modules corresponding to each model type. Data modules help manage data loading,
# transformations, and batching in a consistent way.
from src.data.mae.datamodule import OralMAEDataModule
from src.data.autoencoder.datamodule import OralAutoencoderDataModule
from src.data.vicreg.datamodule import OralVICRegDataModule
from src.data.dino.datamodule import OralDinoDataModule
from src.data.moco.datamodule import OralMOCODataModule
from src.data.classification.datamodule import OralClassificationDataModule
from src.data.contrastive_classification.datamodule import OralContrastiveDataModule
from src.models.classification import OralClassifierModule
from src.models.contrastive_classification import OralContrastiveClassifierModule


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):

    # Directories for experiment outputs
    run_output_dir, checkpoint_dir = get_experiment_dirs(cfg)

    # Setup loggers
    loggers = get_loggers(cfg, run_output_dir)

    # Set precision based on classification mode, for CAE use high precision
    if cfg.classification_mode == 'cae':
        torch.set_float32_matmul_precision("high")
    else: torch.set_float32_matmul_precision('medium')
    
    # Setup the seed for reproducibility
    seed = cfg.get('seed', 42)
    if seed == -1 or seed is None:
        # Use a random seed
        seed = int.from_bytes(os.urandom(4), byteorder="big")
    pl.seed_everything(seed, workers=True)

    # Callbacks list
    callbacks = []
    callbacks.append(get_early_stopping(cfg))
    
    callbacks.append(ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_on_train_epoch_end=True,  
        save_top_k=cfg.get('train.save_top_k', 1),
        monitor="val/loss",  
    ))

    callbacks.append(LossLogCallback(
        log_dir=run_output_dir,
        enable_tensorboard=cfg.get('log.tensorboard', False),
    ))

    # Get the model and data module based on the configuration
    model, data = get_model_and_data(cfg)

    trainer = pl.Trainer(
        default_root_dir= checkpoint_dir,
        logger=loggers,  
        callbacks=callbacks,  
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',  
        devices=cfg.get('train.devices', 1),  
        log_every_n_steps=cfg.get('train.log_every_n_steps', 1),
        max_epochs=cfg.get('train.max_epochs', 100),
    )

    # Start the training loop.
    trainer.fit(model, data)

    classification_mode = cfg.get('classification_mode', 'whole')
    if classification_mode == 'cae':
        output_dir = os.path.join(run_output_dir, 'reconstructed_images')

        # Fetch a batch of images
        imgs, labels, image_id, image_name = next(iter(data.test_dataloader()))

        # Pass images through the autoencoder to obtain reconstructions.
        y_hat = model(imgs)
        from matplotlib import pyplot as plt
        # Loop over each image in the batch.
        for i in range(len(imgs)):
            # Create a figure with two subplots: original vs. reconstructed.
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(imgs[i].permute(1, 2, 0))  # Permute tensor dimensions for plotting.
            ax[0].set_title("Original")
            ax[0].axis("off")
            ax[1].imshow(y_hat[i].detach().permute(1, 2, 0))  # Show the reconstructed image.
            ax[1].set_title("Reconstructed")
            ax[1].axis("off")

            # Ensure the output directory exists and save the figure.
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/{image_name[i]}.jpg")

        # Run testing to evaluate performance on the test set.
        trainer.test(model, data)
        
    elif cfg.classification_mode == 'dino':
        trainer.test(model, data)
    
    elif cfg.classification_mode == 'vicreg':
        trainer.test(model, data)
    
    elif cfg.classification_mode == 'mae':
        # For MAE (Masked Autoencoder): Reconstruct images and save results.
        output_dir = "reconstructe_images/reconstructed_images_mae"
        # Get the first batch including an additional set of images (imgs2).
        imgs, labels, image_id, image_name, imgs2 = next(iter(data.test_dataloader()))
        # Reconstruct images using a specific reconstruction method.
        y_hat = model.reconstructe_images(imgs)
        from matplotlib import pyplot as plt
        for i in range(len(imgs2)):
            fig, ax = plt.subplots(1, 2)
            # Display the original image from the second set (imgs2) after moving it to CPU and adjusting dimensions.
            ax[0].imshow(imgs2[i].cpu().permute(1, 2, 0))
            ax[0].set_title("Original")
            ax[0].axis("off")
            # Display the reconstructed image.
            ax[1].imshow(y_hat[i].cpu().detach().permute(1, 2, 0))
            ax[1].set_title("Reconstructed")
            ax[1].axis("off")

            print(f'Image name: {image_name[i]}')

            # Save the resulting figure to the designated output directory.
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/{image_name[i]}.jpg")

        # Evaluate the MAE model on the test set.
        trainer.test(model, data)
    
    elif cfg.classification_mode == 'moco':
        trainer.test(model, data)
    
    else:
        if cfg.task == 'c' or cfg.task == 'classification':      
            trainer.test(model, datamodule=data)
            if cfg.generate_map == "grad-cam":
                print("--- Generating Saliency Maps ---")
                predictions = trainer.predict(model, datamodule=data)
                predictions = torch.cat(predictions, dim=0)
                predictions = torch.argmax(predictions, dim=1)
                print("Saliency map generation complete.")

def get_model_and_data(cfg: DictConfig) -> Tuple[pl.LightningModule, pl.LightningDataModule]:
    """
    Instantiates and returns the appropriate model and data module based on the
    provided configuration.

    This factory function dynamically discovers class information from the DataModule
    before instantiating the Model, supporting all defined classification modes.
    It is refactored to use dictionaries for module selection to improve
    readability and maintainability.
    """
    # --- Build data paths from config ---
    dataset_dir = os.path.join(
        cfg.dataset.get('base_path'),
        cfg.dataset.get('dataset_name'),
        cfg.dataset.get('augmentation_type')
    )

    # Determine the path suffix based on the specified data format.
    data_format = cfg.dataset.get('data_format')
    path_suffix = '.json' if data_format == 'json' else ''
    if data_format not in ['json', 'directory']:
        raise ValueError(f"Unsupported data_format: {data_format}. Must be 'json' or 'directory'.")

    train_path = os.path.join(dataset_dir, f'train{path_suffix}')
    val_path = os.path.join(dataset_dir, f'val{path_suffix}')
    test_path = os.path.join(dataset_dir, f'test{path_suffix}')

    # Obtain image transformations from a separate utility function.
    train_img_transform, val_img_transform, test_img_transform, img_transform = get_transformations(cfg)

    # --- Instantiate the DataModule using a mapping ---
    DATA_MODULE_MAP = {
        'whole': OralClassificationDataModule,
        'contrastive': OralContrastiveDataModule,
        'cae': OralAutoencoderDataModule,
        'dino': OralDinoDataModule,
        'vicreg': OralVICRegDataModule,
        'mae': OralMAEDataModule,
        'moco': OralMOCODataModule,
    }
    
    classification_mode = cfg.get('classification_mode')
    data_module_class = DATA_MODULE_MAP.get(classification_mode)
    if not data_module_class:
        raise NotImplementedError(f"DataModule for mode '{classification_mode}' is not implemented.")

    # Define common arguments for all DataModule instances.
    data_args = {
        'train': train_path, 
        'val': val_path, 
        'test': test_path,
        'batch_size': cfg.train.get('batch_size'), 
        'num_workers': cfg.train.get('num_workers'),
        'train_transform': train_img_transform, 
        'val_transform': val_img_transform,
        'test_transform': test_img_transform, 
        'transform': img_transform
    }
    data = data_module_class(**data_args)

    # --- Discover dataset properties ---
    # The setup hook is called to allow the DataModule to discover class labels and other metadata.
    data.setup(stage='fit')
    print(f"INFO: Discovered {data.num_classes} classes from the dataset: {data.classes}")
    
    # --- Instantiate the Model using a mapping ---
    task = cfg.get('task')
    if task in ['c', 'classification']:
        # Handle the special case for 'cae' mode, which has a unique constructor signature.
        if classification_mode == 'cae':
            model = Autoencoder(
                cfg.get('ae'),
                cfg.model.get('output_dim'),
                cfg.train.get('lr'),
                cfg.train.get('max_epochs')
            )
        else:
            # Define a mapping for all standard classification models.
            MODEL_MAP = {
                'whole': OralClassifierModule,
                'contrastive': OralContrastiveClassifierModule,
                'dino': OralDinoModule,
                'vicreg': OralVICRegModule,
                'mae': OralMAEModule,
                'moco': OralMOCOModule,
            }
            model_class = MODEL_MAP.get(classification_mode)
            if not model_class:
                raise NotImplementedError(f"Model for mode '{classification_mode}' is not implemented.")

            # Define common arguments for all standard model instances.
            model_args = {
                'weights': cfg.model.get('weights'),
                'num_classes': data.num_classes,  # Dynamically sourced from data module
                'output_dim': cfg.model.get('output_dim'),
                'lr': cfg.train.get('lr'),
                'max_epochs': cfg.train.get('max_epochs')
            }
            model = model_class(**model_args)
            
        # Set the discovered class names on the model instance for logging and other purposes.
        if hasattr(model, 'classes'):
            model.classes = data.classes
    else:
        raise NotImplementedError(f"Task '{task}' is not supported.")
            
    return model, data

if __name__ == "__main__":
    main()