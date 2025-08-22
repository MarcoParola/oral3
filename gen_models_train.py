"""
Training Script for Generative Models.

This script orchestrates the training of various generative models,
selected and configured via a Hydra configuration file. It supports two primary
training workflows:
1.  Fine-tuning of Diffusion Models using Low-Rank Adaptation (LoRA), managed
    by the PyTorch Lightning framework.
2.  Training of StyleGAN models, including variants with multi-class
    classification capabilities (AC-GAN), by wrapping a custom training script.
"""

# --- Standard Library Imports ---
import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple
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
from PIL import Image
import torchvision.transforms as transforms

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
    Verifies the presence of the StyleGAN3 repository and clones it if absent.
    """
    if cfg.model_type != 'stylegan3':
        return True

    repo_path = Path(cfg.get("stylegan3_repo_path", "libs/stylegan3"))
    if not repo_path.is_dir():
        logger.warning(f"StyleGAN3 repository not found at '{repo_path}'. Attempting to clone...")
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        git_url = "https://github.com/franocella/stylegan3.git" # custom Fork
        try:
            subprocess.run(
                ["git", "clone", git_url, str(repo_path)],
                check=True, capture_output=True, text=True
            )
            logger.info("Successfully cloned StyleGAN3 repository.")
        except Exception as e:
            logger.critical(f"Failed to clone StyleGAN3. Please install git or clone it manually. Error: {e}")
            return False
        
    # Add the repository to the system path to enable module imports.
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
    """
    logger.info("--- Starting PyTorch Lightning Training for LoRA/Diffusion Model ---")
    pl.seed_everything(cfg.lora_params.train.get('seed', 42), workers=True)

    # Instantiate the model, dynamically filtering config arguments to match the
    # LoraDiffusionModel constructor's signature for robustness.
    model_init_params = inspect.signature(LoraDiffusionModel.__init__).parameters
    expected_model_args = {name for name, param in model_init_params.items() if name != 'self' and param.kind != inspect.Parameter.VAR_KEYWORD}
    model_args = {k: v for k, v in cfg.lora_params.model.items() if k in expected_model_args}
    model = LoraDiffusionModel(**model_args)
    
    # If resuming, load weights from a specified checkpoint.
    ckpt_path = cfg.lora_params.train.get('resume_from_checkpoint')
    if ckpt_path:
        logger.info(f"Loading weights from checkpoint for resuming: {ckpt_path}")
        try:
            state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)["state_dict"]
            model.load_state_dict(state_dict, strict=False)
            logger.info("Model weights successfully loaded from checkpoint.")
        except Exception as e:
            logger.error(f"Failed to manually load state dict from {ckpt_path}. Error: {e}", exc_info=True)
            return

    # Instantiate datamodule, filtering config arguments
    logger.info("Setting up DiffusionDataModule...")
    dataset_cfg = cfg.lora_params.dataset
    dm_init_params = inspect.signature(DiffusionDataModule.__init__).parameters
    expected_dm_args = {name for name, param in dm_init_params.items() if name != 'self' and param.kind != inspect.Parameter.VAR_KEYWORD}

    datamodule_args = {}

    # Construct absolute paths for data files.
    original_cwd = Path(get_original_cwd())
    base_data_path = original_cwd / dataset_cfg.get("base_path", "data/")

    datamodule_args['train_path'] = str(base_data_path / dataset_cfg.train_file)
    datamodule_args['val_path'] = str(base_data_path / dataset_cfg.val_file)
    datamodule_args['test_path'] = str(base_data_path / dataset_cfg.test_file)

    if 'reg_data' in dataset_cfg:
        datamodule_args['reg_data_path'] = str(base_data_path / dataset_cfg.reg_data)

    # Pass remaining dataset parameters.
    for param_name, param_value in dataset_cfg.items():
        if param_name not in ['base_path', 'train_file', 'val_file', 'test_file', 'reg_data'] and param_name in expected_dm_args:
            datamodule_args[param_name] = param_value
    
    # Pass transform configurations.
    for param in ['base_transform', 'train_transform', 'val_transform', 'test_transform']:
        if param in cfg.lora_params and param in expected_dm_args: 
            datamodule_args[param] = cfg.lora_params[param]

    data = DiffusionDataModule(**datamodule_args)

    # Configure PyTorch Lightning callbacks (e.g., EarlyStopping, ModelCheckpoint).
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    callbacks = []

    if cfg.lora_params.get('early_stopping'):
        es_init_params = inspect.signature(EarlyStopping.__init__).parameters
        expected_es_args = {p for p in es_init_params if p != 'self'}
        early_stopping_args = {k: v for k, v in cfg.lora_params.early_stopping.items() if k in expected_es_args}
        callbacks.append(EarlyStopping(**early_stopping_args))

    checkpoint_cfg = cfg.lora_params.get('checkpoint', {})
    if 'monitor' not in checkpoint_cfg and cfg.lora_params.get('early_stopping'):
        checkpoint_cfg['monitor'] = cfg.lora_params.early_stopping.get('monitor', 'val/epoch_loss')
        
    mc_init_params = inspect.signature(ModelCheckpoint.__init__).parameters
    expected_mc_args = {p for p in mc_init_params if p != 'self'}
    checkpoint_args = {k: v for k, v in checkpoint_cfg.items() if k in expected_mc_args}
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(hydra_output_dir, "checkpoints"), **checkpoint_args)
    callbacks.append(checkpoint_callback)
    
    pl_loggers = [WandbLogger(log_model="best", experiment=wandb_run)] if wandb_run else []

    # Configure and instantiate the PyTorch Lightning Trainer
    trainer_args_raw = OmegaConf.to_container(cfg.lora_params.train, resolve=True)

    # Remove keys not intended for the Trainer constructor.
    for key in ['seed', 'run_test', 'resume_from_checkpoint']:
        trainer_args_raw.pop(key, None)
    trainer_init_params = inspect.signature(pl.Trainer.__init__).parameters
    expected_trainer_args = {p for p in trainer_init_params if p != 'self'}
    trainer_args = {k: v for k, v in trainer_args_raw.items() if k in expected_trainer_args}

    trainer = pl.Trainer(
        default_root_dir=hydra_output_dir,
        callbacks=callbacks,
        logger=pl_loggers if pl_loggers else False,
        **trainer_args
    )

    logger.info("Starting trainer.fit() for LoRA model...")
    trainer.fit(model=model, datamodule=data, ckpt_path=None) # ckpt_path is None because of manually loaded weights
    logger.info("--- LoRA/Diffusion Training Finished ---")

    # Execute the test phase if configured.
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
            pipeline = transforms.Compose([
                transforms.CenterCrop(short_side),
                transforms.Resize((resolution, resolution), antialias=True)
            ])
            processed_img = pipeline(img)
            final_dest_path = dest_path.with_suffix('.jpg')
            final_dest_path.parent.mkdir(parents=True, exist_ok=True)
            processed_img.save(final_dest_path, "JPEG", quality=95)
    except Exception as e:
        logger.warning(f"Could not process image {source_path.name}. Skipping. Error: {e}")


def prepare_stylegan_data_split(
    params: DictConfig, original_cwd: Path, dataset_name: str, data_split: str
) -> Optional[Tuple[Path, int]]:
    """
    Prepares a data split (e.g., 'train', 'val') into the required StyleGAN format
    by processing images and creating a manifest. It returns the path to the
    prepared data directory and the number of classes found.
    """
    # Construct the path to the source data file or directory for the given split
    base_data_path = original_cwd / params.source_data_base_path
    data_file_name = params.get(f'{data_split}_data_file')
    if not data_file_name:
        logger.error(f"'{data_split}_data_file' not defined in config for dataset '{dataset_name}'.")
        return None, 0
    source_path = base_data_path / data_file_name
    
    # Define the destination for the processed data
    prepared_data_root = original_cwd / params.prepared_data_root / f"{dataset_name}_{data_split}"
    resolution = params.get('resolution', 256)

    if not source_path.exists():
        logger.error(f"Source data path for split '{data_split}' not found: {source_path}")
        return None, 0
    
    # If data is already prepared, skip processing but determine num_classes from the manifest
    if prepared_data_root.exists() and (prepared_data_root / "dataset.json").exists():
        logger.info(f"Prepared data for '{dataset_name}' split '{data_split}' already exists. Reading num_classes from manifest.")
        with open(prepared_data_root / "dataset.json", 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        num_classes = len(set(label[1][0] for label in manifest.get("labels", []) if label and len(label) > 1))
        return prepared_data_root, num_classes

    logger.info(f"Preparing data for split '{data_split}' at {resolution}x{resolution}...")
    if prepared_data_root.exists():
        shutil.rmtree(prepared_data_root)
    prepared_data_root.mkdir(parents=True, exist_ok=True)

    all_image_tasks: List[Tuple[Path, Path]] = []

    # Discover images and their corresponding class labels from the source data.
    try:
        if source_path.is_file() and source_path.suffix == '.json':
            # Handle COCO-style JSON manifest.
            image_source_dir = original_cwd / params.coco_image_source_dir
            with open(source_path, 'r', encoding='utf-8') as f: manifest = json.load(f)
            annotations_map = {ann["image_id"]: ann for ann in manifest.get("annotations", [])}
            category_id_to_name = {cat["id"]: cat["name"] for cat in manifest.get("categories", [])}
            for image_info in manifest.get('images', []):
                ann = annotations_map.get(image_info.get("id"))
                if ann and (class_name := category_id_to_name.get(ann.get("category_id"))):
                    src_img_path = image_source_dir / image_info['file_name']
                    dest_img_path = prepared_data_root / class_name / Path(image_info['file_name']).name
                    if src_img_path.exists():
                        all_image_tasks.append((src_img_path, dest_img_path))
        elif source_path.is_dir():
            # Handle directory-based (ImageFolder-style) data.
            img_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
            for src_img_path in source_path.rglob('*'):
                if src_img_path.suffix.lower() in img_exts:
                    class_name = src_img_path.parent.name
                    dest_img_path = prepared_data_root / class_name / src_img_path.name
                    all_image_tasks.append((src_img_path, dest_img_path))
    except Exception as e:
        logger.critical(f"Failed during data discovery for '{data_split}': {e}", exc_info=True); return None, 0
    
    logger.info(f"Found {len(all_image_tasks)} images to process for split '{data_split}'...")
    for src, dest in tqdm(all_image_tasks, desc=f"Processing {data_split} Images"):
        _process_and_save_image(src, dest, resolution)

    logger.info(f"Creating StyleGAN label manifest for '{data_split}'...")
    try:
        all_processed = [p for p in prepared_data_root.rglob('*.jpg')]
        if not all_processed: logger.error(f"No processed images found in '{prepared_data_root}'."); return None, 0
        
        # Determine classes automatically from the folder structure.
        class_names = sorted({p.parent.name for p in all_processed})
        num_classes = len(class_names)
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        logger.info(f"Found {num_classes} classes: {class_names}")

        # The manifest format is a list of [image_path, [class_index]].
        labels_data = [[f"{p.parent.name}/{p.name}", [class_to_idx[p.parent.name]]] for p in all_processed]

        with open(prepared_data_root / "dataset.json", 'w', encoding='utf-8') as f:
            json.dump({"labels": labels_data}, f)
        
        logger.info(f"Successfully created manifest for '{data_split}' with {len(labels_data)} labels.")
        return prepared_data_root, num_classes
    except Exception as e:
        logger.critical(f"Failed to create manifest for '{data_split}': {e}", exc_info=True); return None, 0


def run_stylegan_training(cfg: DictConfig):
    """
    Handles data preparation and dispatches to the bimodal StyleGAN training script.
    """
    logger.info("--- Launching StyleGAN Training Process ---")
    sg3_params = cfg.stylegan3_params
    original_cwd = Path(get_original_cwd())
    sg3_repo_path = original_cwd / cfg.get("stylegan3_repo_path", "libs/stylegan3")
    
    params = OmegaConf.to_container(sg3_params, resolve=True)
    is_ac_gan_mode = params.pop('ac-gan', False)

    # Prepare train and (optional) validation data.
    prepared_train_path, _ = prepare_stylegan_data_split(
        params=sg3_params, original_cwd=original_cwd, dataset_name=cfg.active_dataset, data_split='train'
    )
    if not prepared_train_path:
        logger.critical("Aborting: Training data preparation failed."); return
    
    zip_paths = {}
    zip_paths['train'] = prepared_train_path.with_suffix('.zip')
    
    prepared_val_path = None
    if sg3_params.get('validation_data_file'):
        prepared_val_path, _ = prepare_stylegan_data_split(
            params=sg3_params, original_cwd=original_cwd, dataset_name=cfg.active_dataset, data_split='validation'
        )
        if prepared_val_path:
            zip_paths['val'] = prepared_val_path.with_suffix('.zip')

    # Create .zip files if they don't exist.
    dataset_tool_script = sg3_repo_path / "dataset_tool.py"
    for split, prepared_path in [('train', prepared_train_path), ('val', prepared_val_path)]:
        if not prepared_path: continue
        target_zip = zip_paths[split]
        if not target_zip.exists():
            logger.info(f"Creating dataset ZIP for '{split}' split...")
            resolution_str = f"{sg3_params.resolution}x{sg3_params.resolution}"
            zip_command = [sys.executable, str(dataset_tool_script), f"--source={prepared_path}", f"--dest={target_zip}", f"--resolution={resolution_str}"]
            try:
                subprocess.run(zip_command, check=True, capture_output=True, text=True)
                logger.info(f"Successfully created {split} dataset ZIP: {target_zip}")
            except subprocess.CalledProcessError as e:
                logger.critical(f"Failed to create {split} ZIP. Stderr:\n{e.stderr}"); return
        else:
            logger.info(f"Using existing {split} dataset ZIP: {target_zip}")
    
    # Build and launch the training command.
    custom_train_script = sg3_repo_path / "train.py"
    params['data'] = str(zip_paths['train'])
    if 'val' in zip_paths:
        params['val-data'] = str(zip_paths['val'])
    
    # Clean up keys used only for this master script.
    for key in ['source_data_base_path','coco_image_source_dir', 'image_folder', 'prepared_data_root', 'train_data_file', 'validation_data_file', 'resolution']:
        params.pop(key, None)
    
    if not is_ac_gan_mode:
        for key in ['class-weight', 'val-interval']:
            params.pop(key, None)

    if cfg.log.get('wandb', False):
        params['wandb-log'] = True
        params['wandb-project'] = cfg.wandb.project
        if cfg.wandb.entity:
            params['wandb-entity'] = cfg.wandb.entity
            
    # Build the final command list.
    args = [sys.executable, "-u", str(custom_train_script)]
    
    if is_ac_gan_mode:
        args.append('--ac-gan')

    for key, value in params.items():
        if isinstance(value, bool):
            if value:  
                args.append(f"--{key}")
        elif value is not None:
            args.append(f"--{key}={value}")

    logger.info(f"Executing training command: {' '.join(args)}")
    
    process = subprocess.Popen(args, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()

    if process.returncode == 0:
        logger.info(f"--- StyleGAN Training Finished Successfully (Mode: {'AC-GAN' if is_ac_gan_mode else 'Standard'}) ---")
    else:
        logger.error(f"--- StyleGAN Training FAILED with exit code {process.returncode} (Mode: {'AC-GAN' if is_ac_gan_mode else 'Standard'}) ---")


@hydra.main(version_base=None, config_path="./config", config_name="gen_models_train")
def main(cfg: DictConfig):
    """Main entry point that dispatches to the correct training function."""
    torch.set_float32_matmul_precision('medium')
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    setup_logging(cfg, hydra_output_dir)
    logger.info(f"Hydra Run Output Directory: {hydra_output_dir}")

    # Ensure dependencies are met before proceeding.
    if not setup_stylegan_dependency(cfg):
        sys.exit(1)

    wandb_run = None
    # Initialize Weights & Biases only for Lightning-based models.
    # The StyleGAN subprocess manages its own W&B initialization.
    if cfg.log.get('wandb', False) and cfg.model_type == 'lora_diffusion':
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

    # Execute the selected training workflow and ensure proper cleanup.
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