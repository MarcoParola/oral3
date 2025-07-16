# gen_models_test.py
"""
Evaluation script for generative models (LoRA/Diffusion and StyleGAN3).

Calculates FID, KID, CLIP Score, Precision, Recall, and Coverage for a given
model and dataset configuration.

The script supports two modes:
1. Live Generation: Generates images on the fly.
2. Pre-generated Evaluation: Evaluates existing images from a directory to
   save computation time.

Results are logged to a CSV file for experiment tracking.
"""

import csv
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Third-party Imports
import hydra
import numpy as np
import omegaconf
import prdc
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.utils import save_image

# Local application imports
from src.data.stable_diffusion.datamodule import DiffusionDataModule
from src.data.stable_diffusion.dataset import DiffusionDataset
from src.models.stable_diffusion import LoraDiffusionModel

log = logging.getLogger(__name__)


# --- Generator Wrappers ---

class LoraDiffusionGenerator:
    """A wrapper for LoRA/Diffusion models to standardize image generation."""

    def __init__(self, trained_model: LoraDiffusionModel, mode: str, config: omegaconf.DictConfig):
        if mode not in ['text_to_img', 'img_to_img']:
            raise ValueError(f"Unsupported LoRA generation mode: {mode}")
        self.mode = mode
        self.model = trained_model
        self.diffusion_params = config.diffusion_params
        self.model.freeze()

    def generate(self, batch: dict, device: str = 'cpu', seed: int = None) -> torch.Tensor:
        """Generates and normalizes images based on the configured mode."""
        if self.mode == 'text_to_img':
            generated_images = self.model.generate_samples(
                prompts=batch["prompts"],
                num_inference_steps=self.diffusion_params.diffusion_steps,
                seed=seed
            )
        else:  # img_to_img mode
            guidance_config = omegaconf.OmegaConf.create({
                "generation": {"num_inference_steps": self.diffusion_params.diffusion_steps, "guidance_scale": self.diffusion_params.guidance_scale},
                "image_to_image_settings": {"injection_lambda": self.diffusion_params.injection_lambda}
            })
            generated_images = self.model.generation_with_guidance(
                prompts=batch["prompts"],
                guidance_images=batch["images"],
                cfg=guidance_config,
                seed=seed
            )
        
        # Normalize images to [0, 1] range
        return (generated_images.clamp(-1, 1) + 1) / 2


class StyleGAN3Generator:
    """A wrapper for StyleGAN3 models to standardize image generation."""

    def __init__(self, pkl_path: str, stylegan_params: omegaconf.DictConfig, device: str = 'cpu'):
        self.device = device
        self.params = stylegan_params
        
        try:
            import dnnlib
            import legacy
        except ImportError:
            log.error("Failed to import 'dnnlib' or 'legacy'. Ensure 'stylegan3_project_path' in your config is correct.")
            sys.exit(1)

        log.info(f'Loading G_ema from network pickle "{pkl_path}"...')
        try:
            with dnnlib.util.open_url(pkl_path) as f:
                network_dict = legacy.load_network_pkl(f)
                self.generator = network_dict['G_ema'].to(device)
            
            self.generator.eval()
            self.z_dim = self.generator.z_dim
            self.c_dim = self.generator.c_dim
            log.info("Successfully loaded StyleGAN3 generator.")
        except Exception as e:
            log.error(f"Failed to load StyleGAN3 model from {pkl_path}. Error: {e}", exc_info=True)
            sys.exit(1)

    def generate(self, batch: dict, device: str = 'cpu', seed: int = None, **kwargs) -> torch.Tensor:
        """Generates images from class labels and random latent vectors."""
        class_labels_int = batch["class_labels"].to(self.device)
        class_labels_one_hot = F.one_hot(
            class_labels_int.to(torch.int64), num_classes=self.c_dim
        ).to(torch.float32)

        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        latents = torch.randn([class_labels_one_hot.shape[0], self.z_dim], device=self.device, generator=generator)
        
        generated_images = self.generator(
            z=latents,
            c=class_labels_one_hot,
            truncation_psi=self.params.truncation_psi,
            noise_mode=self.params.noise_mode,
            **kwargs 
        )
        # Normalize images to [0, 1] range
        return (generated_images.clamp(-1, 1) + 1) / 2


# --- Evaluation Pipeline ---

class ModelEvaluator(pl.LightningModule):
    """A PyTorch Lightning Module to orchestrate the model evaluation process."""
    
    def __init__(self, config: omegaconf.DictConfig, csv_log_path: Path, model_type: str, dataset_name: str,
                 image_save_path: Optional[Path], generator=None, 
                 pregenerated_image_paths: Optional[List[Path]] = None, 
                 idx_to_class_name: Optional[dict] = None):
        super().__init__()

        if not (generator or pregenerated_image_paths):
            raise ValueError("Must provide either a 'generator' or 'pregenerated_image_paths'.")
        if generator and pregenerated_image_paths:
            raise ValueError("Cannot use 'generator' and 'pregenerated_image_paths' simultaneously.")

        self.generator = generator
        self.pregenerated_image_paths = pregenerated_image_paths
        self.config = config
        self.csv_log_path = csv_log_path
        self.image_save_path = image_save_path
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.idx_to_class_name = idx_to_class_name
        self.model_is_diffusion = isinstance(self.generator, LoraDiffusionGenerator)
        self.image_counter = 0

        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            from torchmetrics.image.kid import KernelInceptionDistance
            from torchmetrics.multimodal.clip_score import CLIPScore
        except ImportError:
            log.error("Torchmetrics is not installed correctly. Please run: pip install 'torchmetrics[image,multimodal]'")
            sys.exit(1)

        self.fid = FrechetInceptionDistance(feature=2048)
        self.clip_score = CLIPScore(model_name_or_path='openai/clip-vit-base-patch32')
        
        kid_default_subset_size = self.config.get('kid_subset_size', 50)
        self.kid = KernelInceptionDistance(subset_size=kid_default_subset_size)
        
        # Buffers to manually store features for PRDC computation.
        self.real_feature_buffer = []
        self.fake_feature_buffer = []

        image_size = self.config.get('image_size', 256)
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def _load_image_batch(self, image_paths: List[Path]) -> torch.Tensor:
        images = [self.image_transform(Image.open(path).convert("RGB")) for path in image_paths]
        return torch.stack(images)

    def on_test_start(self):
        """Called at the beginning of the test loop."""
        # Move the Inception feature extractor to the correct device.
        self.fid.inception.to(self.device)

        if self.generator and self.image_save_path:
            log.info(f"Generated images will be saved to: {self.image_save_path}")
            self.image_save_path.mkdir(parents=True, exist_ok=True)
        else:
            log.info("Using pre-generated images for evaluation. No new images will be saved.")

    def test_step(self, batch: dict, batch_index: int):
        real_images = batch["images"]

        if self.pregenerated_image_paths:
            batch_size = real_images.shape[0]
            start_idx = batch_index * batch_size
            end_idx = start_idx + batch_size
            
            if start_idx >= len(self.pregenerated_image_paths):
                return
            
            current_image_paths = self.pregenerated_image_paths[start_idx:end_idx]
            fake_images = self._load_image_batch(current_image_paths).to(self.device)
        else:
            # A progressive seed ensures diversity in generation across batches.
            base_seed = self.config.get('seed', 0)
            progressive_seed = base_seed + batch_index
            fake_images = self.generator.generate(batch, device=self.device, seed=progressive_seed)

            if self.image_save_path:
                class_indices = batch["class_labels"].cpu().numpy()
                class_names = [self.idx_to_class_name.get(idx, f"class_{idx}") for idx in class_indices]
                for i in range(fake_images.size(0)):
                    safe_class_name = class_names[i].replace(" ", "_").lower()
                    filename = f"{safe_class_name}_{self.image_counter:05d}.png"
                    save_image(fake_images[i], self.image_save_path / filename)
                    self.image_counter += 1

        # The InceptionV3 model expects uint8 images in [0, 255].
        real_images_for_inception = (real_images.clamp(0, 1) * 255).byte()
        fake_images_for_inception = (fake_images.clamp(0, 1) * 255).byte()

        # Update metrics that handle their own feature extraction.
        self.fid.update(real_images_for_inception, real=True)
        self.fid.update(fake_images_for_inception, real=False)
        self.kid.update(real_images_for_inception, real=True)
        self.kid.update(fake_images_for_inception, real=False)
        
        # Manually extract and store features for PRDC.
        real_feats = self.fid.inception(real_images_for_inception)
        fake_feats = self.fid.inception(fake_images_for_inception)
        self.real_feature_buffer.append(real_feats.cpu())
        self.fake_feature_buffer.append(fake_feats.cpu())
        
        if self.model_is_diffusion or self.pregenerated_image_paths:
            prompts = batch["prompts"]
        else:  # StyleGAN case requires constructing prompts from class labels.
            class_indices_prompt = batch["class_labels"].cpu().numpy()
            class_names_prompt = [self.idx_to_class_name[idx].lower() for idx in class_indices_prompt]
            clip_prompts_map = self.config.get('clip_prompts_by_class', {})
            prompts = [clip_prompts_map.get(name, name) for name in class_names_prompt]

        self.clip_score.update(fake_images_for_inception, prompts)

    def on_test_epoch_end(self):
        """Computes, logs, and saves final metrics at the end of the test epoch."""
        log.info("Data accumulation complete. Computing final metrics...")
        
        fid_score = self.fid.compute()
        clip_score = self.clip_score.compute()

        # Dynamically adjust KID subset_size to prevent errors with small datasets.
        num_samples = min(
            sum(f.shape[0] for f in self.kid.real_features),
            sum(f.shape[0] for f in self.kid.fake_features)
        )
        
        if self.kid.subset_size >= num_samples:
            if num_samples > 1:
                new_size = num_samples - 1
                log.warning(
                    f"KID subset_size ({self.kid.subset_size}) is too large for the number of samples ({num_samples}). "
                    f"Adjusting to {new_size} to prevent an error."
                )
                self.kid.subset_size = new_size
            else:
                self.kid.subset_size = 0  # Mark as invalid for computation.

        if self.kid.subset_size > 0:
            kid_score, kid_std = self.kid.compute()
        else:
            log.warning("Cannot compute KID due to insufficient samples (< 2). Returning NaN.")
            kid_score = torch.tensor(float('nan'), device=self.device)
            kid_std = torch.tensor(float('nan'), device=self.device)

        # Compute PRDC using the manually collected features.
        real_features = torch.cat(self.real_feature_buffer, dim=0).numpy()
        fake_features = torch.cat(self.fake_feature_buffer, dim=0).numpy()
        
        k_neighbors = 5
        if real_features.shape[0] < k_neighbors or fake_features.shape[0] < k_neighbors:
            log.warning(f"Not enough features for PRDC (real: {real_features.shape[0]}, fake: {fake_features.shape[0]}). Skipping.")
            prdc_scores = {'precision': np.nan, 'recall': np.nan, 'coverage': np.nan}
        else:
            log.info(f"Calculating PRDC with {real_features.shape[0]} real and {fake_features.shape[0]} fake features...")
            prdc_scores = prdc.compute_prdc(
                real_features=real_features,
                fake_features=fake_features,
                nearest_k=k_neighbors
            )

        model_mode_details = ""
        if isinstance(self.generator, LoraDiffusionGenerator):
            mode = self.generator.mode
            if mode == 'img_to_img':
                lambda_val = self.generator.diffusion_params.injection_lambda
                model_mode_details = f" (Mode: {mode}, λ={lambda_val:.2f})"
            else:
                model_mode_details = f" (Mode: {mode})"
        elif self.pregenerated_image_paths:
            model_mode_details = " (Mode: Pre-generated)"

        log.info(f"\n--- Metrics Summary for {self.model_type.upper()}{model_mode_details} on {self.dataset_name} ---")
        log.info(f"Fréchet Inception Distance (FID): {fid_score:.5f}")
        log.info(f"Kernel Inception Distance (KID): {kid_score:.5f} ± {kid_std:.5f}")
        log.info(f"CLIP Score: {clip_score:.5f}")
        log.info(f"Precision: {prdc_scores.get('precision', np.nan):.5f}")
        log.info(f"Recall: {prdc_scores.get('recall', np.nan):.5f}")
        log.info(f"Coverage: {prdc_scores.get('coverage', np.nan):.5f}")
        log.info("--------------------------------------------------")
        
        self._write_metrics_to_csv(fid_score, clip_score, kid_score, kid_std, prdc_scores)
        
    def _write_metrics_to_csv(self, fid_score: torch.Tensor, clip_score: torch.Tensor, 
                              kid_score: torch.Tensor, kid_std: torch.Tensor, 
                              prdc_scores: dict):
        """Appends the computed metrics and run metadata to a CSV file."""
        if not self.csv_log_path:
            log.warning("'csv_log_path' not provided. Skipping saving metrics to file.")
            return

        try:
            self.csv_log_path.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = "N/A (pre-generated)"
            if self.generator:
                checkpoint_path = self.config.model_checkpoints[self.model_type][self.dataset_name]

            dataset_splits = "+".join(self.config.dataset_splits)
            file_exists = self.csv_log_path.is_file()
            metrics_record = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_name': self.model_type,
                'dataset_name': self.dataset_name,
                'dataset_splits': dataset_splits,
                'checkpoint_path': checkpoint_path,
                'fid_score': f"{fid_score.item():.5f}",
                'clip_score': f"{clip_score.item():.5f}",
                'kid_score': f"{kid_score.item():.5f}",
                'kid_std': f"{kid_std.item():.5f}",
                'precision': f"{prdc_scores.get('precision', np.nan):.5f}",
                'recall': f"{prdc_scores.get('recall', np.nan):.5f}",
                'coverage': f"{prdc_scores.get('coverage', np.nan):.5f}",
            }
            with open(self.csv_log_path, 'a', newline='', encoding='utf-8') as csv_file:
                fieldnames = metrics_record.keys()
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(metrics_record)
            log.info(f"Successfully saved metrics to {self.csv_log_path}")
        except Exception as e:
            log.error(f"Failed to save metrics to {self.csv_log_path}. Error: {e}", exc_info=True)


# --- Main Execution ---

@hydra.main(version_base=None, config_path="config", config_name="gen_models_test")
def main(config: omegaconf.DictConfig):
    
    pl.seed_everything(config.get('seed', 42), workers=True)
    
    model_type = config.active_model_type
    dataset_name = config.active_dataset
    log.info(f"Active model type: '{model_type}', Dataset: '{dataset_name}'")

    run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_output_dir = Path(f"logs/test/{model_type}/{dataset_name}/{run_timestamp}")
    run_output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Run output will be saved in: {run_output_dir}")
    
    if model_type == 'stylegan3':
        sg3_project_path = config.stylegan3_params.get('stylegan3_project_path')
        if sg3_project_path and Path(sg3_project_path).is_dir():
            sys.path.append(str(sg3_project_path))
            log.info(f"Added StyleGAN3 project to sys.path: {sg3_project_path}")

    use_gpu = config.devices > 0 and torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    log.info(f"Execution device: {device.upper()}")
    
    dataset_config = config.datasets[dataset_name]
    datamodule = DiffusionDataModule(
        train_path=str(Path(dataset_config.base_path) / dataset_config.train_file),
        val_path=str(Path(dataset_config.base_path) / dataset_config.val_file),
        test_path=str(Path(dataset_config.base_path) / dataset_config.test_file),
        coco_image_subdir=dataset_config.get('image_folder'),
        batch_size=config.batch_size, num_workers=config.num_workers,
        return_class_label=True 
    )
    datamodule.setup(stage='test')

    datasets_to_combine = [getattr(datamodule, f"{split}_dataset") for split in config.dataset_splits if hasattr(datamodule, f"{split}_dataset")]
    if not datasets_to_combine: 
        raise ValueError("No valid dataset splits were provided or the datasets are empty.")
    
    evaluation_dataset = ConcatDataset(datasets_to_combine)
    
    # Enable persistent workers for performance if multiple workers are used.
    use_persistent_workers = True if config.num_workers > 0 else False
    evaluation_dataloader = DataLoader(
        evaluation_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        collate_fn=DiffusionDataset.collate_fn,
        persistent_workers=use_persistent_workers
    )
    
    accelerator = "gpu" if use_gpu else "cpu"
    lightning_trainer = pl.Trainer(accelerator=accelerator, devices=config.devices if use_gpu else 1, logger=False, benchmark=True)
    torch.set_float32_matmul_precision('high')
    
    csv_log_path = run_output_dir / "metrics_summary.csv"
    pregenerated_images_dir = config.get("generated_images_path")
    pregenerated_image_paths = []

    if pregenerated_images_dir:
        image_dir = Path(pregenerated_images_dir)
        if image_dir.is_dir():
            pregenerated_image_paths = sorted(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')))
            if not pregenerated_image_paths:
                log.warning(f"Directory '{image_dir}' is empty. Proceeding with live generation.")
            else:
                if len(pregenerated_image_paths) != len(evaluation_dataset):
                    log.warning(f"Mismatch: {len(pregenerated_image_paths)} pre-generated vs. {len(evaluation_dataset)} real images.")
                log.info(f"Found {len(pregenerated_image_paths)} pre-generated images. Skipping generation.")
        else:
            log.warning(f"Directory '{image_dir}' not found. Proceeding with live generation.")

    if pregenerated_image_paths:
        model_evaluator = ModelEvaluator(
            config=config,
            csv_log_path=csv_log_path,
            model_type=model_type,
            dataset_name=dataset_name,
            pregenerated_image_paths=pregenerated_image_paths,
            image_save_path=None
        )
    else:
        # Live generation setup
        try:
            checkpoint_path_str = config.model_checkpoints[model_type][dataset_name]
            checkpoint_path = Path(checkpoint_path_str)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            log.info(f"Loading checkpoint: {checkpoint_path}")
        except KeyError:
            raise KeyError(f"Checkpoint path for model '{model_type}' on dataset '{dataset_name}' not defined in config.")
        
        generator = None
        idx_to_class_name = {v: k for k, v in datamodule.test_dataset.class_to_idx.items()}

        if model_type == 'lora_diffusion':
            trained_model = LoraDiffusionModel.load_from_checkpoint(checkpoint_path, map_location=device, strict=False)
            mode = config.diffusion_params.diffusion_mode
            if mode == 'img_to_img' and config.diffusion_params.injection_lambda is None:
                config.diffusion_params.injection_lambda = random.choice([0.7, 0.8])
            log.info(f"Starting test execution (Mode: {mode})")
            generator = LoraDiffusionGenerator(trained_model, mode=mode, config=config)

        elif model_type == 'stylegan3':
            log.info("Starting test execution (Mode: conditional generation)")
            generator = StyleGAN3Generator(str(checkpoint_path), stylegan_params=config.stylegan3_params, device=device)
        else:
            raise ValueError(f"Unsupported 'active_model_type' in configuration: {model_type}")

        model_evaluator = ModelEvaluator(
            config=config,
            csv_log_path=csv_log_path,
            model_type=model_type,
            dataset_name=dataset_name,
            generator=generator,
            idx_to_class_name=idx_to_class_name,
            image_save_path=(run_output_dir / "generated_images") 
        )

    lightning_trainer.test(model=model_evaluator, dataloaders=evaluation_dataloader)
    log.info("Evaluation process completed successfully.")

if __name__ == "__main__":
    main()