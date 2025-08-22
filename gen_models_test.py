"""
Script for Evaluating Generative Models on Pre-generated Images.

This script calculates standard image generation quality metrics, including
Fréchet Inception Distance (FID), Kernel Inception Distance (KID), CLIP Score,
and Precision, Recall, and Coverage (PRDC).

It operates by comparing a directory of pre-generated synthetic images against a
corresponding set of real images from a reference dataset. The results are
logged to a CSV file for systematic experiment tracking.
"""

# --- Standard Library Imports ---
import csv
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# --- Third-party Imports ---
import hydra
import numpy as np
import omegaconf
import prdc
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms
from rich.console import Console
from rich.table import Table

# Import Torchmetrics modules for standard metric calculations.
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchmetrics.multimodal.clip_score import CLIPScore
except ImportError:
    log.error("Torchmetrics is not installed correctly. Please run: pip install 'torchmetrics[image,multimodal]'")
    sys.exit(1)

# --- Local Application Imports ---
from src.data.stable_diffusion.datamodule import DiffusionDataModule
from src.data.stable_diffusion.dataset import DiffusionDataset

# Configure a module-level logger.
log = logging.getLogger(__name__)


# --- Evaluation Pipeline ---

class ModelEvaluator(pl.LightningModule):
    """
    A PyTorch Lightning Module to orchestrate the model evaluation process
    using pre-generated images.
    """
    
    def __init__(self, config: omegaconf.DictConfig, csv_log_path: Path, model_type: str, dataset_name: str,
                 pregenerated_image_paths: List[Path]):
        super().__init__()

        self.pregenerated_image_paths = pregenerated_image_paths
        self.config = config
        self.csv_log_path = csv_log_path
        self.model_type = model_type
        self.dataset_name = dataset_name

        # Initialize metric objects. FID and KID use an InceptionV3 network with 2048-dim features.
        self.fid = FrechetInceptionDistance(feature=2048)
        self.clip_score = CLIPScore(model_name_or_path='openai/clip-vit-base-patch32')
        
        kid_default_subset_size = self.config.get('kid_subset_size', 50)
        self.kid = KernelInceptionDistance(subset_size=kid_default_subset_size)
        
        # Initialize buffers to manually store features for PRDC computation.
        # This is required as PRDC is not yet integrated into Torchmetrics.
        self.real_feature_buffer = []
        self.fake_feature_buffer = []

        # Define a standardized image transformation for loading pre-generated images.
        image_size = self.config.get('image_size', 256)
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    
    def _load_image_batch(self, image_paths: List[Path]) -> torch.Tensor:
        """Loads and transforms a batch of images from a list of file paths."""
        images = [self.image_transform(Image.open(path).convert("RGB")) for path in image_paths]
        return torch.stack(images)

    def on_test_start(self):
        """Prepares the evaluation environment at the beginning of the test loop."""
        # The Inception feature extractor must be explicitly moved to the correct device.
        self.fid.inception.to(self.device)

    def test_step(self, batch: dict, batch_index: int):
        """Performs a single evaluation step for a batch of data."""
        real_images = batch["images"]
        batch_size = real_images.shape[0]

        # Load the corresponding batch of pre-generated fake images.
        start_idx = batch_index * batch_size
        end_idx = start_idx + batch_size
        
        if start_idx >= len(self.pregenerated_image_paths):
            return
        
        current_image_paths = self.pregenerated_image_paths[start_idx:end_idx]
        fake_images = self._load_image_batch(current_image_paths).to(self.device)

        # Convert images to uint8 in the [0, 255] range, as expected by the InceptionV3 model.
        real_images_for_inception = (real_images.clamp(0, 1) * 255).byte()
        fake_images_for_inception = (fake_images.clamp(0, 1) * 255).byte()

        # Update metrics that manage their own feature extraction.
        self.fid.update(real_images_for_inception, real=True)
        self.fid.update(fake_images_for_inception, real=False)
        self.kid.update(real_images_for_inception, real=True)
        self.kid.update(fake_images_for_inception, real=False)
        
        # Manually extract and store Inception features for subsequent PRDC calculation.
        real_feats = self.fid.inception(real_images_for_inception)
        fake_feats = self.fid.inception(fake_images_for_inception)
        self.real_feature_buffer.append(real_feats.cpu())
        self.fake_feature_buffer.append(fake_feats.cpu())
        
        # Update CLIP score with the fake images and their corresponding text prompts.
        prompts = batch["prompts"]
        self.clip_score.update(fake_images_for_inception, prompts)

    def on_test_epoch_end(self):
        """Computes, logs, and saves all final metrics at the end of the test epoch."""
        log.info("Data accumulation complete. Computing final metrics...")
        
        fid_score = self.fid.compute()
        clip_score = self.clip_score.compute()

        # Dynamically adjust KID subset_size to prevent errors if the number of
        # samples is smaller than the configured subset size.
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

        # Compute PRDC metrics using the manually collected feature embeddings.
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

        self._pretty_print_metrics(fid_score, kid_score, kid_std, clip_score, prdc_scores)

        
        self._write_metrics_to_csv(fid_score, clip_score, kid_score, kid_std, prdc_scores)
        
    def _write_metrics_to_csv(self, fid_score: torch.Tensor, clip_score: torch.Tensor, 
                              kid_score: torch.Tensor, kid_std: torch.Tensor, 
                              prdc_scores: dict):
        """Appends the computed metrics and run metadata to a CSV log file."""
        if not self.csv_log_path:
            log.warning("'csv_log_path' not provided. Skipping saving metrics to file.")
            return

        try:
            self.csv_log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine the source of the pre-generated images for record-keeping.
            checkpoint_path = self.config.get("generated_images_path", "N/A (path not specified)")

            dataset_splits = "+".join(self.config.dataset_splits)
            file_exists = self.csv_log_path.is_file()
            metrics_record = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_name': self.model_type,
                'dataset_name': self.dataset_name,
                'dataset_splits': dataset_splits,
                'images_source_path': checkpoint_path,
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

    def _pretty_print_metrics(self, fid_score, kid_score, kid_std, clip_score, prdc_scores):

        console = Console()
        table = Table(title=f"Metrics Summary for {self.model_type.upper()} on {self.dataset_name}")

        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        fmt = lambda v: np.format_float_positional(float(v), precision=5, trim='-')

        table.add_row("FID", fmt(fid_score))
        table.add_row("KID", f"{fmt(kid_score)}")
        table.add_row("CLIP Score", fmt(clip_score))
        table.add_row("Precision", fmt(prdc_scores.get('precision', np.nan)))
        table.add_row("Recall", fmt(prdc_scores.get('recall', np.nan)))
        table.add_row("Coverage", fmt(prdc_scores.get('coverage', np.nan)))

        console.print(table)

# --- Main Execution ---

@hydra.main(version_base=None, config_path="config", config_name="gen_models_test")
def main(config: omegaconf.DictConfig):
    
    # Configure and seed the execution environment for reproducibility.
    pl.seed_everything(config.get('seed', 42), workers=True)
    
    model_type = config.active_model_type
    dataset_name = config.active_dataset
    log.info(f"Active model type: '{model_type}', Dataset: '{dataset_name}'")

    # Define a unique output directory for the current evaluation run.
    run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_output_dir = Path(f"logs/gen_test/{model_type}/{dataset_name}/{run_timestamp}")
    run_output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Run output will be saved in: {run_output_dir}")
    
    use_gpu = config.devices > 0 and torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    log.info(f"Execution device: {device.upper()}")
    
    # Configure the data module to load the real reference images.
    dataset_config = config.datasets[dataset_name]
    datamodule = DiffusionDataModule(
        train_path=str(Path(dataset_config.base_path) / dataset_config.train_file),
        val_path=str(Path(dataset_config.base_path) / dataset_config.val_file),
        test_path=str(Path(dataset_config.base_path) / dataset_config.test_file),
        image_folder=dataset_config.get('image_folder'),
        batch_size=config.batch_size, num_workers=config.num_workers,
        return_class_label=True 
    )
    datamodule.setup()

    # Create a single evaluation dataset by concatenating the specified splits.
    datasets_to_combine = [getattr(datamodule, f"{split}_dataset") for split in config.dataset_splits if hasattr(datamodule, f"{split}_dataset") and getattr(datamodule, f"{split}_dataset") is not None]

    if not datasets_to_combine: 
        raise ValueError("No valid dataset splits were provided or the datasets are empty.")
    
    evaluation_dataset = ConcatDataset(datasets_to_combine)
    
    evaluation_dataloader = DataLoader(
        evaluation_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        collate_fn=DiffusionDataset.collate_fn,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    # Initialize the PyTorch Lightning Trainer.
    accelerator = "gpu" if use_gpu else "cpu"
    lightning_trainer = pl.Trainer(accelerator=accelerator, devices=config.devices if use_gpu else 1, logger=False, benchmark=True)
    torch.set_float32_matmul_precision('high')
    
    csv_log_path = run_output_dir / "metrics_summary.csv"
    pregenerated_images_dir = config.get("generated_images_path")
    
    # Discover and validate the paths of pre-generated images.
    if not pregenerated_images_dir or not Path(pregenerated_images_dir).is_dir():
        raise FileNotFoundError(f"The required 'generated_images_path' is not specified or does not exist: {pregenerated_images_dir}")

    image_dir = Path(pregenerated_images_dir)
    image_extensions = ('*.png', '*.jpg', '*.jpeg')
    all_pregenerated_paths = []
    for ext in image_extensions:
        all_pregenerated_paths.extend(image_dir.rglob(ext))
    
    if not all_pregenerated_paths:
        raise FileNotFoundError(f"The specified directory '{image_dir}' contains no images.")

    # Subsample the larger dataset (real or fake) to ensure both have the same
    # number of samples for a fair comparison.
    n_real_initial = len(evaluation_dataset)
    n_fake_initial = len(all_pregenerated_paths)

    if n_real_initial != n_fake_initial:
        log.warning(f"Mismatch: {n_fake_initial} pre-generated vs. {n_real_initial} real images.")
        min_size = min(n_real_initial, n_fake_initial)
        log.info(f"Randomly subsampling the larger set to match the smaller size: {min_size}")

        if n_fake_initial > min_size:
            pregenerated_image_paths = sorted(random.sample(all_pregenerated_paths, min_size))
        else:
            pregenerated_image_paths = sorted(all_pregenerated_paths)

        if n_real_initial > min_size:
            indices = random.sample(range(n_real_initial), min_size)
            evaluation_dataset = Subset(evaluation_dataset, indices)
    else:
        pregenerated_image_paths = sorted(all_pregenerated_paths)

    log.info(f"Final evaluation size: {len(pregenerated_image_paths)} fake vs. {len(evaluation_dataset)} real images.")

    # Instantiate the model evaluator with the prepared data paths.
    model_evaluator = ModelEvaluator(
        config=config,
        csv_log_path=csv_log_path,
        model_type=model_type,
        dataset_name=dataset_name,
        pregenerated_image_paths=pregenerated_image_paths
    )

    # Execute the evaluation process.
    lightning_trainer.test(model=model_evaluator, dataloaders=evaluation_dataloader)
    log.info("Evaluation process completed successfully.")

if __name__ == "__main__":
    main()