# augmentation_train.py
# -*- coding: utf-8 -*-
"""
Training Script for Generative Models.

This script serves as a master controller for training different types of
generative models, dispatched via a configuration file. It supports:
1.  Fine-tuning Diffusion Models with LoRA (via PyTorch Lightning).
2.  Training StyleGAN3 models (by wrapping the official NVIDIA training script).

The script handles all necessary setup, including dependency checks, logging,
data preparation, and dispatching to the appropriate training function based on
the `model_type` specified in the configuration.
"""

# --- Standard Library Imports ---
import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import Optional
import shutil
from tqdm import tqdm
import json
import inspect

# --- Third-party Imports ---
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
from hydra.utils import get_original_cwd
from typing import Optional, List, Tuple

import torchvision.transforms as transforms
from PIL import Image

# --- Local Application Imports ---
try:
    from src.models.stable_diffusion import LoraDiffusionModel
    from src.data.stable_diffusion.datamodule import DiffusionDataModule
except ImportError as e:
    print(f"ERROR: Could not import project modules: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)


def setup_stylegan_dependency(cfg: DictConfig) -> bool:
    """
    Checks for the StyleGAN3 repository and clones it if not present.
    Returns True on success, False on failure.
    """
    if cfg.model_type != 'stylegan3':
        return True

    repo_path = Path(cfg.get("stylegan3_repo_path", "libs/stylegan3"))
    if not repo_path.is_dir():
        logger.warning(f"StyleGAN3 repository not found at '{repo_path}'. Attempting to clone...")
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        git_url = "https://github.com/franocella/stylegan3.git"
        try:
            subprocess.run(
                ["git", "clone", git_url, str(repo_path)],
                check=True, capture_output=True, text=True
            )
            logger.info("Successfully cloned StyleGAN3 repository.")
        except Exception as e:
            logger.critical(f"Failed to clone StyleGAN3. Please install git or clone it manually. Error: {e}")
            return False

    if str(repo_path) not in sys.path:
        sys.path.append(str(repo_path))
    logger.info("StyleGAN3 dependency is ready.")
    return True


def setup_logging(cfg: DictConfig, output_dir: str):
    """Configures the root Python logger."""
    log_level = cfg.log.get("level", "INFO").upper()
    log_file = os.path.join(output_dir, 'train.log')

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger.info(f"Logging configured. Log file at: {log_file}")


def run_lightning_training(cfg: DictConfig, wandb_run: Optional[wandb.sdk.wandb_run.Run]):
    """
    Handles the training process for PyTorch Lightning-based models like LoRA/Diffusion.

    Args:
        cfg (DictConfig): The Hydra configuration object containing all parameters.
        wandb_run (Optional[wandb.sdk.wandb_run.Run]): The Weights & Biases run object, if active.
    """
    logger.info("--- Starting PyTorch Lightning Training for LoRA/Diffusion Model ---")

    # Set random seed for reproducibility
    pl.seed_everything(cfg.lora_params.train.get('seed', 42), workers=True)

    # --- Model Instantiation and Parameter Filtering ---
    # Get the expected parameters for LoraDiffusionModel's __init__
    # We inspect the signature to find all arguments except 'self' and 'kwargs'.
    model_init_params = inspect.signature(LoraDiffusionModel.__init__).parameters
    expected_model_args = {
        name for name, param in model_init_params.items()
        if name != 'self' and param.kind != inspect.Parameter.VAR_KEYWORD
    }

    # Filter cfg.lora_params.model to only include expected arguments
    model_args = {}
    for key, value in cfg.lora_params.model.items():
        if key in expected_model_args:
            model_args[key] = value
        else:
            logger.warning(
                f"Parameter '{key}' found in 'cfg.lora_params.model' is not an expected argument "
                f"for LoraDiffusionModel.__init__ and will be ignored."
            )

    # Instantiate the model with filtered arguments
    model = LoraDiffusionModel(**model_args)
    
    # --- Checkpoint Loading for Resuming Training ---
    ckpt_path = cfg.lora_params.train.get('resume_from_checkpoint')
    if ckpt_path:
        logger.info(f"Loading weights from checkpoint for resuming: {ckpt_path}")
        try:
            # Manually load the state dictionary from the checkpoint file.
            # We add `weights_only=False` to allow loading checkpoints saved with
            # PyTorch Lightning, which contain non-tensor objects like OmegaConf configs.
            state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)["state_dict"]
            model.load_state_dict(state_dict, strict=False)
            logger.info("Model weights successfully loaded from checkpoint.")
        except Exception as e:
            logger.error(f"Failed to manually load state dict from {ckpt_path}. Error: {e}", exc_info=True)
            return

    # --- DataModule Instantiation and Parameter Filtering ---
    logger.info("Setting up DiffusionDataModule...")
    dataset_cfg = cfg.lora_params.dataset

    # Get the expected parameters for DiffusionDataModule's __init__
    dm_init_params = inspect.signature(DiffusionDataModule.__init__).parameters
    expected_dm_args = {
        name for name, param in dm_init_params.items()
        if name != 'self' and param.kind != inspect.Parameter.VAR_KEYWORD
    }
    
    datamodule_args = {}
    original_cwd = Path(get_original_cwd())
    base_data_path = original_cwd / dataset_cfg.get("base_path", "data/")
    
    # Path arguments, should always be present or handled by DataModule defaults
    datamodule_args['train_path'] = str(base_data_path / dataset_cfg.train_data)
    datamodule_args['val_path'] = str(base_data_path / dataset_cfg.val_data)
    datamodule_args['test_path'] = str(base_data_path / dataset_cfg.test_data)
    if 'reg_data' in dataset_cfg:
        datamodule_args['reg_data_path'] = str(base_data_path / dataset_cfg.reg_data)

    # Filter dataset_cfg parameters
    for param_name, param_value in dataset_cfg.items():
        if param_name not in ['base_path', 'train_data', 'val_data', 'test_data', 'reg_data']:
            if param_name in expected_dm_args:
                datamodule_args[param_name] = param_value
            else:
                logger.warning(
                    f"Parameter '{param_name}' found in 'cfg.lora_params.dataset' is not an expected argument "
                    f"for DiffusionDataModule.__init__ and will be ignored."
                )

    for param in ['base_transform', 'train_transform', 'val_transform', 'test_transform']:
        if param in cfg.lora_params and param in expected_dm_args: 
            datamodule_args[param] = cfg.lora_params[param]
        elif param in cfg.lora_params:
             logger.warning(
                f"Parameter '{param}' found in 'cfg.lora_params' is not an expected argument "
                f"for DiffusionDataModule.__init__ and will be ignored."
            )

    data = DiffusionDataModule(**datamodule_args)

    # --- Callbacks, Loggers, and Trainer Setup ---
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    callbacks = []

    # EarlyStopping Callback
    if cfg.lora_params.get('early_stopping'):
        # Filter early_stopping parameters
        es_init_params = inspect.signature(EarlyStopping.__init__).parameters
        expected_es_args = {
            name for name, param in es_init_params.items()
            if name != 'self' and param.kind != inspect.Parameter.VAR_KEYWORD
        }
        early_stopping_args = {}
        for key, value in cfg.lora_params.early_stopping.items():
            if key in expected_es_args:
                early_stopping_args[key] = value
            else:
                logger.warning(
                    f"Parameter '{key}' found in 'cfg.lora_params.early_stopping' is not an expected argument "
                    f"for EarlyStopping.__init__ and will be ignored."
                )
        callbacks.append(EarlyStopping(**early_stopping_args))

    # ModelCheckpoint Callback
    checkpoint_cfg = cfg.lora_params.get('checkpoint', {})
    if 'monitor' not in checkpoint_cfg and cfg.lora_params.get('early_stopping'):
        checkpoint_cfg['monitor'] = cfg.lora_params.early_stopping.get('monitor', 'val/epoch_loss')

    # Filter checkpoint parameters
    mc_init_params = inspect.signature(ModelCheckpoint.__init__).parameters
    expected_mc_args = {
        name for name, param in mc_init_params.items()
        if name != 'self' and param.kind != inspect.Parameter.VAR_KEYWORD
    }
    checkpoint_args = {}
    for key, value in checkpoint_cfg.items():
        if key in expected_mc_args:
            checkpoint_args[key] = value
        else:
            logger.warning(
                f"Parameter '{key}' found in 'cfg.lora_params.checkpoint' is not an expected argument "
                f"for ModelCheckpoint.__init__ and will be ignored."
            )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(hydra_output_dir, "checkpoints"),
        **checkpoint_args
    )
    callbacks.append(checkpoint_callback)
    
    pl_loggers = [WandbLogger(log_model="all", experiment=wandb_run)] if wandb_run else []

    # --- Trainer Instantiation ---
    trainer_args_raw = OmegaConf.to_container(cfg.lora_params.train, resolve=True)
    # Remove known "non-trainer" arguments from the training config
    for key in ['seed', 'run_test', 'resume_from_checkpoint']:
        trainer_args_raw.pop(key, None)

    # Get the expected parameters for pl.Trainer's __init__
    trainer_init_params = inspect.signature(pl.Trainer.__init__).parameters
    expected_trainer_args = {
        name for name, param in trainer_init_params.items()
        if name != 'self' and param.kind != inspect.Parameter.VAR_KEYWORD # Exclude self and **kwargs
    }
    
    trainer_args = {}
    for key, value in trainer_args_raw.items():
        if key in expected_trainer_args:
            trainer_args[key] = value
        else:
            logger.warning(
                f"Parameter '{key}' found in 'cfg.lora_params.train' is not an expected argument "
                f"for pl.Trainer.__init__ and will be ignored."
            )

    trainer = pl.Trainer(
        default_root_dir=hydra_output_dir,
        callbacks=callbacks,
        logger=pl_loggers if pl_loggers else False, # Use False to disable default logger if no PL loggers
        **trainer_args
    )

    logger.info("Starting trainer.fit() for LoRA model...")
    # Pass ckpt_path=None because we already handled loading the model's weights.
    trainer.fit(model=model, datamodule=data, ckpt_path=None)
    logger.info("--- LoRA/Diffusion Training Finished ---")

    # --- Testing Phase ---
    if cfg.lora_params.train.get('run_test', False):
        logger.info("--- Starting Testing Phase ---")
        best_ckpt_path = checkpoint_callback.best_model_path
        if best_ckpt_path and os.path.exists(best_ckpt_path):
            trainer.test(model=model, datamodule=data, ckpt_path=best_ckpt_path)
        else:
            logger.warning("Could not find best checkpoint. Skipping test phase.")



def _process_and_save_image(source_path: Path, dest_path: Path, resolution: int):
    """
    Opens an image, center-crops and resizes it, and saves it as a high-quality JPG.
    """
    try:
        with Image.open(source_path) as img:
            img = img.convert("RGB")
            
            short_side = min(img.size)
            transform_pipeline = transforms.Compose([
                transforms.CenterCrop(short_side),
                transforms.Resize((resolution, resolution), antialias=True)
            ])
            processed_img = transform_pipeline(img)

            final_dest_path = dest_path.with_suffix('.jpg')
            
            final_dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            processed_img.save(final_dest_path, "JPEG", quality=95)
            
    except Exception as e:
        logger.warning(f"Could not process image {source_path.name}. Skipping. Error: {e}")


def prepare_stylegan_source_data(params: DictConfig, original_cwd: Path, dataset_name: str) -> Optional[Path]:
    """
    Prepares raw source data into the required StyleGAN3 format by processing
    images and creating a manifest file.
    """
    source_path = original_cwd / params.source_data_path
    prepared_data_root = original_cwd / params.prepared_data_root
    resolution = params.get('resolution', 256)

    if not source_path.exists():
        logger.error(f"Source data path not found: {source_path}")
        return None
    
    reorganized_dir = prepared_data_root
    logger.info(f"Source: '{source_path}' -> Prepared data destination: '{reorganized_dir}'")

    if reorganized_dir.exists() and (reorganized_dir / "dataset.json").exists():
        logger.info(f"Prepared data for '{dataset_name}' already exists. Skipping.")
        return reorganized_dir

    logger.info(f"Preparing and processing data for '{dataset_name}' at {resolution}x{resolution}...")
    if reorganized_dir.exists():
        shutil.rmtree(reorganized_dir)
    reorganized_dir.mkdir(parents=True, exist_ok=True)

    all_image_tasks: List[Tuple[Path, Path]] = []

    try:
        if source_path.is_file() and source_path.suffix == '.json':
            logger.info("Source is a COCO JSON file. Gathering image paths...")
            image_source_dir = original_cwd / params.coco_image_source_dir
            if not image_source_dir.is_dir():
                logger.error(f"Required 'coco_image_source_dir' not found: {image_source_dir}")
                return None

            with open(source_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            annotations_map = {ann["image_id"]: ann for ann in manifest.get("annotations", [])}
            category_id_to_name = {cat["id"]: cat["name"] for cat in manifest.get("categories", [])}

            for image_info in manifest.get('images', []):
                annotation = annotations_map.get(image_info.get("id"))
                if annotation and (class_name := category_id_to_name.get(annotation.get("category_id"))):
                    source_image_path = image_source_dir / image_info['file_name']
                    dest_image_path = reorganized_dir / class_name / Path(image_info['file_name']).name
                    if source_image_path.exists():
                        all_image_tasks.append((source_image_path, dest_image_path))

        elif source_path.is_dir():
            logger.info("Source is a directory. Gathering image paths...")
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
            for source_image_path in source_path.rglob('*'):
                if source_image_path.suffix.lower() in image_extensions:
                    class_name = source_image_path.parent.name
                    dest_image_path = reorganized_dir / class_name / source_image_path.name
                    all_image_tasks.append((source_image_path, dest_image_path))
        else:
            logger.error(f"Unrecognized source data type for StyleGAN3: {source_path}")
            return None
    except Exception as e:
        logger.critical(f"Failed during data discovery step: {e}", exc_info=True)
        return None

    logger.info(f"Found {len(all_image_tasks)} total images to process...")
    for src, dest in tqdm(all_image_tasks, desc="Processing and Saving Images"):
        _process_and_save_image(src, dest, resolution)

    logger.info(f"Creating StyleGAN3 label manifest in '{reorganized_dir}'...")
    try:
        all_processed_images = [p for p in reorganized_dir.rglob('*') if p.suffix.lower() == '.jpg']
        if not all_processed_images:
            logger.error(f"No processed .jpg images found in '{reorganized_dir}'."); return None
        
        class_names = sorted(list(set([p.parent.name for p in all_processed_images])))
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        labels_data = []
        for img_path in all_processed_images:
            class_name = img_path.parent.name
            relative_path = f"{class_name}/{img_path.name}"
            labels_data.append([relative_path, class_to_idx[class_name]])

        with open(reorganized_dir / "dataset.json", 'w', encoding='utf-8') as f:
            json.dump({"labels": labels_data}, f)

        logger.info(f"Successfully created manifest with {len(labels_data)} labels.")
        return reorganized_dir
    except Exception as e:
        logger.critical(f"Failed to create 'dataset.json' manifest: {e}", exc_info=True)
        return None

def run_stylegan_training(cfg: DictConfig):
    """
    Handles the data preparation and training for StyleGAN3.

    This pipeline prepares the dataset into the format required by the official
    NVIDIA script (a labeled .zip file) and then launches that script as an
    external subprocess with the correct parameters.

    Args:
        cfg: The global configuration object from Hydra.
    """
    logger.info("--- Launching External StyleGAN3 Training Process ---")
    sg3_params = cfg.stylegan3_params
    
    # Get the original CWD to resolve all paths correctly, making it robust to Hydra's CWD changes.
    original_cwd = Path(get_original_cwd())

    # Step 1: Prepare the source data into a standardized directory format.
    prepared_data_path = prepare_stylegan_source_data(
        params=sg3_params,
        original_cwd=original_cwd,
        dataset_name=cfg.active_dataset
    )
    if not prepared_data_path:
        logger.critical("Aborting StyleGAN training due to data preparation failure.")
        return

    # Step 2: Create the final .zip dataset file using StyleGAN3's dataset_tool.py.
    sg3_repo_path = original_cwd / cfg.get("stylegan3_repo_path", "libs/stylegan3")
    target_zip_path = prepared_data_path.with_suffix('.zip')
    
    if not target_zip_path.exists():
        logger.info(f"Creating labeled dataset ZIP from source: '{prepared_data_path}'...")
        dataset_tool_script = sg3_repo_path / "dataset_tool.py"
        
        resolution = sg3_params.get('resolution', 256)
        resolution_str = f"{resolution}x{resolution}"

        # Build the command using sys.executable for robustness.
        zip_command = [
            sys.executable, str(dataset_tool_script),
            f"--source={prepared_data_path}",
            f"--dest={target_zip_path}",
            f"--resolution={resolution_str}"
        ]

        logger.info(f"Applying pre-transform: cropping to square and resizing to {resolution_str}...")

        try:
            subprocess.run(zip_command, check=True)
            logger.info(f"Successfully created dataset ZIP: {target_zip_path}")
            
        except subprocess.CalledProcessError:
            # Update the error message as we are no longer capturing stderr.
            # The actual error from the script will be visible in the console above.
            logger.critical(f"Failed to create dataset ZIP file. See console output above for details.")
            return
    else:
        logger.info(f"Using existing dataset ZIP: {target_zip_path}")
    
    # Step 3: Launch the StyleGAN3 training script with the prepared .zip dataset.
    sg3_train_script = sg3_repo_path / "train.py"
    params_for_training = OmegaConf.to_container(sg3_params, resolve=True)
    params_for_training['data'] = str(target_zip_path)

    # Remove parameters that are only for data preparation, not for the training script itself.
    keys_to_remove = [
        'source_data_path', 'coco_image_source_dir', 'prepared_data_root', 'resolution'
    ]
    for key in keys_to_remove:
        params_for_training.pop(key, None)

    args = [sys.executable, "-W", "ignore", str(sg3_train_script)]

    for key, value in params_for_training.items():
        if value is not None:
            value_str = str(value).lower() if isinstance(value, bool) else str(value)
            args.append(f"--{key}={value_str}")

    logger.info(f"Executing training command: {' '.join(args)}")
    
    # Launch the training process, allowing its output to stream to the console.
    process = subprocess.Popen(args, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()

    if process.returncode == 0:
        logger.info("--- StyleGAN3 Training Process Finished Successfully ---")
    else:
        logger.error(f"--- StyleGAN3 Training Process FAILED with exit code {process.returncode} ---")

@hydra.main(version_base=None, config_path="./config", config_name="augmentation_train.yaml")
def main(cfg: DictConfig):
    """Main entry point that dispatches to the correct training function."""
    
    torch.set_float32_matmul_precision('medium')

    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    setup_logging(cfg, hydra_output_dir)
    logger.info(f"Hydra Run Output Directory: {hydra_output_dir}")

    if not setup_stylegan_dependency(cfg):
        sys.exit(1)

    wandb_run = None
    if cfg.log.get('wandb', False):
        try:
            logger.info("Initializing Weights & Biases (WandB)...")
            hyperparameters = OmegaConf.to_container(cfg, resolve=True)
            wandb_run = wandb.init(
                dir=hydra_output_dir, 
                entity=cfg.wandb.get('entity'), 
                project=cfg.wandb.get('project'), 
                name=cfg.wandb.get('run_name'), 
                config=hyperparameters
            )
        except Exception as e:
            logger.error(f"Error initializing WandB: {e}")

    # Dispatch to the correct training function based on model_type
    try:
        if cfg.model_type == 'lora_diffusion':
            run_lightning_training(cfg, wandb_run)
        elif cfg.model_type == 'stylegan3':
            run_stylegan_training(cfg)
        else:
            logger.error(f"Invalid 'model_type' in config: '{cfg.model_type}'")
            sys.exit(1)
    finally:
        if wandb_run:
            wandb.finish()
            logger.info("WandB run finished.")


if __name__ == "__main__":
    main()