import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb 

# Import utility functions
from pathlib import Path
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

# Import data modules corresponding to each model type. 
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

    # Setup temporary directory for wandb in the current working directory
    os.environ['WANDB_DIR'] = '.'

    # Set precision based on classification mode, for CAE use high precision
    if cfg.classification_mode == 'cae':
        torch.set_float32_matmul_precision("high")
    else: torch.set_float32_matmul_precision('medium')
    
    # Setup the seed for reproducibility
    seed = cfg.train.seed
    if seed == -1 or seed is None:
        # Use a random seed
        seed = int.from_bytes(os.urandom(4), byteorder="big")
    pl.seed_everything(seed, workers=True)

    print("The seed has been set for reproducibility and is:", seed)

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
    default_root_dir=checkpoint_dir,
    logger=loggers,
    callbacks=callbacks,
    accelerator='cuda' if torch.cuda.is_available() else 'cpu',
    devices=cfg.train.devices,
    log_every_n_steps=cfg.train.log_every_n_steps,
    max_epochs=cfg.train.max_epochs,
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
    Factory function to instantiate the correct model and data module.
    """
    dataset_dir = Path(cfg.dataset.base_path)

    dataset_type = cfg.dataset.type
    path_suffix = '.json' if dataset_type == 'json' else ''
    if dataset_type not in ['json', 'directory']:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Must be 'json' or 'directory'.")

    train_path = dataset_dir / f'train{path_suffix}'
    val_path = dataset_dir / f'val{path_suffix}'
    test_path = dataset_dir / f'test{path_suffix}'

    train_img_transform, val_img_transform, test_img_transform, img_transform = get_transformations(cfg)

    DATA_MODULE_MAP = {
        'whole': OralClassificationDataModule,
        'contrastive': OralContrastiveDataModule,
        'cae': OralAutoencoderDataModule,
        'dino': OralDinoDataModule,
        'vicreg': OralVICRegDataModule,
        'mae': OralMAEDataModule,
        'moco': OralMOCODataModule,
    }
    
    classification_mode = cfg.classification_mode
    data_module_class = DATA_MODULE_MAP.get(classification_mode)
    if not data_module_class:
        raise NotImplementedError(f"DataModule for mode '{classification_mode}' is not implemented.")

    data_args = {
        'train_path': str(train_path), 
        'val_path': str(val_path), 
        'test_path': str(test_path),
        'batch_size': cfg.train.batch_size, 
        'num_workers': cfg.train.num_workers,
        'train_transform': train_img_transform, 
        'val_transform': val_img_transform,
        'test_transform': test_img_transform, 
        'transform': img_transform,
        'train_aug_path': cfg.dataset.get('augmentation_data_path', None),
        'train_aug_percentage': cfg.dataset.get('augmentation_percentage', None)
    }
    
    data = data_module_class(**data_args)

    data.setup(stage='fit')
    
    if cfg.task == 'classification':
        if classification_mode == 'cae':
            model = Autoencoder(
                cfg.ae, cfg.model.output_dim, cfg.train.lr, cfg.train.max_epochs
            )
        else:
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

            model_args = {
                'weights': cfg.model.weights,
                'num_classes': data.num_classes,
                'output_dim': cfg.model.output_dim,
                'lr': cfg.train.lr,
                'max_epochs': cfg.train.max_epochs
            }
            model = model_class(**model_args)
            
        if hasattr(model, 'classes'):
            model.classes = data.classes
    else:
        raise NotImplementedError(f"Task '{cfg.task}' is not supported.")
            
    return model, data


if __name__ == "__main__":
    main()