# -*- coding: utf-8 -*-
"""
LoraDiffusionModel: A PyTorch Lightning module for LoRA/QLoRA fine-tuning
of text-to-image diffusion models.
"""

# --- Standard Library Imports ---
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Third-party Imports ---
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.utils import make_grid

# --- Optional Dependency Imports with Robust Handling ---
try:
    from peft import LoraConfig, get_peft_model, PeftModel
    # Added for seamless saving/loading of consolidated weights
    from diffusers import StableDiffusionPipeline
    # Import safetensors for direct .safetensors saving
    import safetensors.torch
except ImportError as e:
    logging.critical("PEFT library not found. Please install with: pip install peft")
    raise e

try:
    from torchmetrics import MeanMetric
    from torchmetrics.multimodal import CLIPScore
    from torchmetrics.image.fid import FrechetInceptionDistance as FID
    from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as LPIPS
except ImportError as e:
    logging.critical(
        "torchmetrics not found. Metrics will fail. "
        "Run: pip install torchmetrics[image,multimodal] lpips"
    )
    raise e

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None
    logging.warning(
        "bitsandbytes not found. 8-bit optimizer and 4-bit quantization are unavailable."
    )

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x  # Fallback to a no-op if tqdm is not installed
    logging.info("tqdm not found. Progress bars during inference will be disabled.")


# --- Diffusers & Transformers Imports ---
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    DDIMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer, BitsAndBytesConfig

# --- WandB Import ---
try:
    import wandb
except ImportError:
    wandb = None
    logging.info("wandb library not found. WandB logging disabled. Run: pip install wandb")

logger = logging.getLogger(__name__)

__all__ = ["LoraDiffusionModel"]


# ==============================================================================
# LoraDiffusionModel Lightning Module (Final Version)
# ==============================================================================
class LoraDiffusionModel(pl.LightningModule):
    """PyTorch Lightning module for LoRA/QLoRA fine-tuning of Stable Diffusion.

    This module handles the setup of diffusion components, LoRA adapter injection,
    quantization (QLoRA), and the training/validation loop with configurable
    metrics and logging.
    """

    def __init__(
        self,
        # --- Model Identification & Paths ---
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
        vae_name_or_path: Optional[str] = None, # Optional path to a custom VAE, ortherwise uses the one from the main model.
        # --- LoRA Configuration ---
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        unet_lora_target_modules: Tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0"),
        train_text_encoder_lora: bool = True,
        text_encoder_lora_target_modules: Tuple[str, ...] = ("q_proj", "v_proj"),
        text_encoder_lora_rank: Optional[int] = None, # Defaults to `lora_rank` if not specified.
        text_encoder_lora_alpha: Optional[int] = None, # Defaults to `lora_alpha` if not specified.
        # --- Training Hyperparameters ---
        learning_rate: float = 1e-4,
        text_encoder_lora_lr_scale: float = 1.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        use_8bit_adam: bool = False,
        # --- Hardware & Optimization ---
        use_4bit_quantization: bool = False,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_compute_dtype: str = "bfloat16",
        mixed_precision: str = "bf16",
        gradient_checkpointing: bool = True,
        # --- Validation & Logging ---
        validation_metric: str = "LPIPS",
        clip_model_name_or_path: str = "openai/clip-vit-large-patch14",
        lpips_net_type: str = "alex",
        fid_feature_size: int = 2048,
        num_validation_images: int = 4,
        val_num_inference_steps: int = 25,
        log_every_n_epochs: int = 1,
        image_size: int = 256,
        validation_seed: Optional[int] = 42,
    ):
        """Initializes the LoraDiffusionModel.

        Args:
            pretrained_model_name_or_path: Path to pretrained model or model identifier.
            vae_name_or_path: Optional path to a VAE; if None, uses the one from the main model.
            lora_rank: The rank 'r' of the LoRA update matrices.
            lora_alpha: The scaling factor for LoRA updates.
            lora_dropout: Dropout probability for LoRA layers.
            unet_lora_target_modules: Tuple of UNet module names to apply LoRA to.
            train_text_encoder_lora: Whether to apply LoRA to the text encoder.
            text_encoder_lora_target_modules: Tuple of text encoder module names to apply LoRA to.
            text_encoder_lora_rank: Optional rank for text encoder LoRA. Defaults to `lora_rank`.
            text_encoder_lora_alpha: Optional alpha for text encoder LoRA. Defaults to `lora_alpha`.
            learning_rate: The base learning rate for the optimizer.
            text_encoder_lora_lr_scale: LR scaling factor for the text encoder's LoRA params.
            adam_beta1: Beta1 for Adam optimizer.
            adam_beta2: Beta2 for Adam optimizer.
            adam_weight_decay: Weight decay for Adam optimizer.
            adam_epsilon: Epsilon for Adam optimizer.
            use_8bit_adam: Whether to use the 8-bit Adam optimizer.
            use_4bit_quantization: Whether to use 4-bit QLoRA quantization.
            bnb_4bit_quant_type: Quantization type for 4-bit (`"nf4"` or `"fp4"`).
            bnb_4bit_compute_dtype: Compute dtype for 4-bit quantization.
            mixed_precision: Training precision (`"bf16"`, `"fp16"`, or `"fp32"`).
            gradient_checkpointing: Whether to use gradient checkpointing.
            validation_metric: Metric for validation (`"LPIPS"`, `"CLIPScore"`, `"FID"`, or `"none"`).
            clip_model_name_or_path: Model name for CLIPScore calculation.
            lpips_net_type: Network type for LPIPS (`"vgg"` or `"alex"`).
            fid_feature_size: Feature dimension for FID (e.g., 2048 for InceptionV3).
            num_validation_images: Number of images to generate for validation.
            val_num_inference_steps: Number of DDIM steps for validation image generation.
            log_every_n_epochs: Frequency of logging validation images and metrics.
            image_size: The size (height and width) of the images.
            validation_seed: A fixed seed for reproducible validation image generation. If None, uses a random seed.
        """
        super().__init__()
        # PyTorch Lightning automatically saves hyperparameters passed to __init__
        self.save_hyperparameters()

        # Configure precision based on hparams
        self.torch_dtype = self._configure_precision()

        # Configure 4-bit quantization for QLoRA if enabled
        bnb_config = self._configure_quantization()

        # Load core diffusion model components
        self._load_core_components(bnb_config)
        # Freeze base model weights
        self._freeze_models()
        # Apply LoRA adapters to UNet and optionally Text Encoder
        self._apply_lora()

        # Enable gradient checkpointing for memory efficiency if specified
        if self.hparams.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.hparams.train_text_encoder_lora:
                self.text_encoder.gradient_checkpointing_enable()

        # Initialize training and validation metrics
        self._initialize_metrics()
        # List to store validation outputs for epoch end processing
        self._validation_outputs = []

        logger.info(f"LoraDiffusionModel initialized. Validation metric: {self.active_val_metric_name.upper()}")

    def _configure_precision(self) -> torch.dtype:
        """Determines the appropriate torch dtype based on the mixed_precision hyperparameter.

        Returns:
            torch.dtype: The configured torch data type.
        """
        precision = self.hparams.mixed_precision.lower()
        if "bf16" in precision:
            return torch.bfloat16
        if "16" in precision:
            return torch.float16
        return torch.float32
    

    def _configure_quantization(self) -> Optional[BitsAndBytesConfig]:
        """Configures 4-bit quantization (QLoRA) using BitsAndBytesConfig if enabled.

        Returns:
            Optional[BitsAndBytesConfig]: The quantization configuration or None if not used.
        """
        if not self.hparams.use_4bit_quantization:
            return None

        # Check if bitsandbytes is available before trying to use it
        if bnb is None:
            logger.error(
                "bitsandbytes is not installed, but use_4bit_quantization=True. QLoRA is disabled."
            )
            return None # Explicitly return None if bnb is not available

        logger.info(
            f"Configuring 4-bit quantization (QLoRA) with compute_dtype={self.hparams.bnb_4bit_compute_dtype}"
        )

        compute_dtype_str = self.hparams.bnb_4bit_compute_dtype
        if compute_dtype_str == "bf16":
            compute_dtype = torch.bfloat16
        elif compute_dtype_str == "fp16":
            compute_dtype = torch.float16
        else:
            # Fallback for other string values, e.g., "float32"
            compute_dtype = getattr(torch, compute_dtype_str)

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.hparams.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    
    
    def _load_core_components(self, bnb_config: Optional[BitsAndBytesConfig]):
        """Loads the VAE, Tokenizer, Schedulers, UNet, and Text Encoder."""
        model_path = self.hparams.pretrained_model_name_or_path

        vae_path = self.hparams.vae_name_or_path or model_path
        # VAE is typically kept in float32 for stability, or its original dtype.
        # Ensure it's loaded appropriately.
        self.vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if self.hparams.vae_name_or_path is None else None,
            torch_dtype=torch.float32 # VAE usually benefits from full precision
        )

        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
        self.inference_scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)

        # load_kwargs will be passed to from_pretrained.
        # If bnb_config is present, torch_dtype should be None to let bitsandbytes handle it.
        # Otherwise, use self.torch_dtype.
        load_kwargs = {
            "quantization_config": bnb_config,
            "torch_dtype": None if bnb_config else self.torch_dtype,
        }
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder", **load_kwargs
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            model_path, subfolder="unet", **load_kwargs
        )

    def _freeze_models(self):
        """Freezes the parameters of the VAE, UNet, and Text Encoder base models.
        Only LoRA adapters will be trainable.
        """
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        logger.info("Froze VAE, UNet, and Text Encoder base weights.")

    def _apply_lora(self):
        """Applies LoRA adapters to the UNet and optionally to the Text Encoder.
        This modifies the models in-place, making only the LoRA adapters trainable.
        """
        # Configure and apply LoRA to UNet
        unet_lora_config = LoraConfig(
            r=self.hparams.lora_rank,
            lora_alpha=self.hparams.lora_alpha,
            lora_dropout=self.hparams.lora_dropout,
            target_modules=list(self.hparams.unet_lora_target_modules),
            bias="none", # LoRA typically does not modify biases
        )
        self.unet = get_peft_model(self.unet, unet_lora_config)
        self.unet.print_trainable_parameters()

        # Optionally configure and apply LoRA to Text Encoder
        if self.hparams.train_text_encoder_lora:
            te_lora_config = LoraConfig(
                r=self.hparams.text_encoder_lora_rank or self.hparams.lora_rank,
                lora_alpha=self.hparams.text_encoder_lora_alpha or self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                target_modules=list(self.hparams.text_encoder_lora_target_modules),
                bias="none",
            )
            self.text_encoder = get_peft_model(self.text_encoder, te_lora_config)
            self.text_encoder.print_trainable_parameters()

    def _initialize_metrics(self):
        """Initializes training loss metric and the selected validation metric.
        Handles dynamic metric loading based on `validation_metric` hyperparameter.
        """
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.active_val_metric = None
        self.active_val_metric_name = "none"

        # Map metric names to their classes and initialization arguments
        metric_map = {
            # For LPIPS, if 'normalize=True' is passed to the constructor,
            # the metric expects inputs in [-1, 1] range and handles its own normalization.
            # If your images are already normalized to [0, 1] before updating,
            # then 'normalize=False' should be used here, and inputs should be [0, 1].
            # Given the conversion to [0,1] in _update_and_log_metric, set normalize=False.
            "lpips": (LPIPS, {"net_type": self.hparams.lpips_net_type, "normalize": False}),
            "clipscore": (CLIPScore, {"model_name_or_path": self.hparams.clip_model_name_or_path}),
            # For FID, 'normalize=True' means the metric handles normalization internally
            # (often to [0, 255] uint8 if it's the default expectation).
            # The manual conversion to uint8 in _update_and_log_metric aligns well with this.
            "fid": (FID, {"feature": self.hparams.fid_feature_size, "normalize": True}),
        }

        metric_key = self.hparams.validation_metric.lower().strip()
        if metric_key in metric_map:
            metric_class, kwargs = metric_map[metric_key]
            try:
                self.active_val_metric = metric_class(**kwargs)
                self.active_val_metric_name = metric_key
            except Exception as e:
                logger.error(f"Failed to initialize metric '{metric_key}': {e}. Validation metric disabled.")

    @torch.no_grad()
    def _encode_vae(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encodes pixel values to latents using the VAE.

        The VAE encoding operation is performed in float32 for numerical stability,
        as recommended for VAEs in diffusion models.

        Args:
            pixel_values: Input pixel values, typically in range [-1, 1].

        Returns:
            torch.Tensor: Encoded latent representation.
        """
        latents = self.vae.encode(pixel_values.to(dtype=torch.float32)).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    @torch.no_grad()
    def _decode_vae(self, latents: torch.Tensor) -> torch.Tensor:
        """Decodes latents back to pixel values using the VAE.

        Args:
            latents: Latent representation to decode.

        Returns:
            torch.Tensor: Decoded image pixels in range [-1, 1].
        """
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents.to(self.vae.dtype)).sample
        return image.clamp(-1, 1)

    def _encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """Encodes a list of text prompts into text embeddings using the CLIPTextModel.

        Args:
            text_prompts: A list of string prompts.

        Returns:
            torch.Tensor: Encoded text embeddings.
        """
        text_inputs = self.tokenizer(
            text_prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        text_embeddings = self.text_encoder(input_ids=input_ids)[0]
        return text_embeddings

    def _shared_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Performs a single forward pass for training or validation and computes the MSE loss.

        Args:
            batch: A dictionary containing input data, typically with 'pixel_values' and 'text'.

        Returns:
            torch.Tensor: The mean squared error loss.
        """
        # Encode images to latents
        # Ensure latents are in the correct dtype for UNet. Autocast typically handles this.
        latents = self._encode_vae(batch["pixel_values"]).to(self.torch_dtype)
        
        # Sample noise for the diffusion process
        noise = torch.randn_like(latents)
        # Sample random timesteps
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=self.device,
        ).long()

        # Add noise to latents based on timesteps
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Encode text prompts to embeddings.
        # _encode_text should return embeddings in a compatible dtype.
        # Autocast in the UNet forward pass will handle casting if needed.
        encoder_hidden_states = self._encode_text(batch["text"])
        
        # Predict noise using UNet.
        # The autocast context in training_step/validation_step (handled by Lightning)
        # will wrap the UNet call and manage precision for compute-intensive ops.
        # It's important to keep noise_pred and noise in float for MSE loss calculation
        # to avoid potential precision issues or overflows with lower precision dtypes.
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """The main training loop logic for a single batch.

        Args:
            batch: The current batch of training data.
            batch_idx: The index of the current batch.

        Returns:
            torch.Tensor: The computed loss for the current training step.
        """
        loss = self._shared_step(batch)
        self.train_loss.update(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """The main validation loop logic for a single batch.

        Args:
            batch: The current batch of validation data.
            batch_idx: The index of the current batch.
        """
        loss = self._shared_step(batch)
        self.val_loss.update(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Store prompts and original images from the first batch for epoch-end logging
        if batch_idx == 0:
            self._validation_outputs.append(
                {"prompts": batch["text"], "original_images": batch["pixel_values"]}
            )

    def on_validation_epoch_end(self):
        """Generates images, computes metrics, and logs them at the end of the validation epoch.
        This method is called by PyTorch Lightning automatically.
        """
        # Skip if no validation outputs were collected (e.g., validation disabled or no batches)
        if not self._validation_outputs:
            logger.info("No validation outputs to process. Skipping on_validation_epoch_end.")
            return

        # Determine if visuals and metrics should be logged based on frequency and metric availability
        # We also check if active_val_metric is not None to avoid errors if metric initialization failed.
        should_log_visuals = (
            self.active_val_metric is not None
            and (self.current_epoch + 1) % self.hparams.log_every_n_epochs == 0
        )

        if not should_log_visuals:
            # Clear stored outputs if no logging will occur to free memory
            self._validation_outputs.clear()
            logger.info(f"Skipping image and metric logging for epoch {self.current_epoch+1} based on frequency or metric availability.")
            return

        # Retrieve data from the first validation batch for logging
        batch_data = self._validation_outputs[0]
        prompts = batch_data["prompts"]
        original_images = batch_data["original_images"]

        # Select a subset of images/prompts for logging
        num_images_to_log = min(len(prompts), self.hparams.num_validation_images)
        log_prompts = prompts[:num_images_to_log]
        log_originals = original_images[:num_images_to_log]

        logger.info(f"Generating {num_images_to_log} samples for validation at epoch {self.current_epoch+1}.")
        # Generate images based on the selected prompts
        generated_images = self.generate_samples(
            prompts=log_prompts,
            num_inference_steps=self.hparams.val_num_inference_steps,
        )

        if generated_images is not None and generated_images.numel() > 0:
            # Update and log the active validation metric
            self._update_and_log_metric(generated_images, log_originals, log_prompts)
            # Log generated images to WandB if enabled
            self._log_images_to_wandb(generated_images.cpu(), log_prompts)
        else:
            logger.warning("No images generated for validation logging.")

        # Clear stored outputs for the next validation epoch
        self._validation_outputs.clear()
        logger.info(f"Finished on_validation_epoch_end for epoch {self.current_epoch+1}.")


    def _update_and_log_metric(self, pred_images: torch.Tensor, real_images: torch.Tensor, prompts: List[str]):
        """Updates the active validation metric with the generated and real data, then logs the result.

        Args:
            pred_images: Tensor of generated images.
            real_images: Tensor of real images from the dataset.
            prompts: List of text prompts used for generation.
        """
        # Ensure the metric is on the correct device
        self.active_val_metric.to(self.device)

        # Normalize and update metric based on type
        if self.active_val_metric_name == "lpips":
            # LPIPS expects images in [0, 1] if normalize=False was set in constructor
            pred_norm = (pred_images + 1) / 2
            real_norm = (real_images + 1) / 2
            self.active_val_metric.update(pred_norm, real_norm)
            logger.debug("LPIPS metric updated with [0,1] normalized images.")
        elif self.active_val_metric_name in ["clipscore", "fid"]:
            # CLIPScore and FID often expect uint8 images [0, 255]
            # Convert images from [-1, 1] float to [0, 255] uint8
            pred_uint8 = ((pred_images / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
            
            if self.active_val_metric_name == "clipscore":
                self.active_val_metric.update(pred_uint8, prompts)
                logger.debug("CLIPScore metric updated with [0,255] uint8 images.")
            else:  # FID
                real_uint8 = ((real_images / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
                self.active_val_metric.update(real_uint8, real=True)
                self.active_val_metric.update(pred_uint8, real=False)
                logger.debug("FID metric updated with [0,255] uint8 images.")

        # Compute and log the metric score, then reset the metric state
        try:
            score = self.active_val_metric.compute()
            self.log(f"val/{self.active_val_metric_name}", score, on_epoch=True, prog_bar=True, sync_dist=True)
            logger.info(f"Logged val/{self.active_val_metric_name}: {score.item():.4f}")
        except Exception as e:
            logger.error(f"Error computing or logging metric {self.active_val_metric_name}: {e}")
        self.active_val_metric.reset()


    def _log_images_to_wandb(self, images: torch.Tensor, prompts: List[str]):
        """Logs a grid of generated images to WandB, if the WandbLogger is configured.

        Args:
            images: Tensor of images to log.
            prompts: List of corresponding prompts for image captions.
        """
        # Check if WandbLogger is active and wandb library is available
        if not isinstance(self.logger, WandbLogger) or wandb is None:
            return

        try:
            # Create a grid of images for visualization
            grid = make_grid(images, normalize=True)
            caption = " | ".join(prompts)
            self.logger.experiment.log(
                {f"val/generated_samples": wandb.Image(grid, caption=caption)},
                step=self.global_step,
            )
            logger.info(f"Logged generated samples to WandB for epoch {self.current_epoch+1}.")
        except Exception as e:
            logger.error(f"Failed to log images to WandB: {e}")

    @torch.no_grad()
    def generate_samples(
        self, prompts: List[str], num_inference_steps: int
    ) -> torch.Tensor:
        """Generates image samples from a list of prompts using the DDIM scheduler.

        Args:
            prompts: A list of text prompts for image generation.
            num_inference_steps: The number of inference steps for the DDIM scheduler.

        Returns:
            torch.Tensor: A tensor of generated images.
        """
        if not prompts:
            logger.warning("generate_samples called with empty prompts. Returning empty tensor.")
            return torch.tensor([])

        # Set models to evaluation mode
        self.unet.eval()
        self.text_encoder.eval()
        self.vae.eval()

        generator = None
        # Use a fixed seed for reproducible validation generation if specified
        if self.hparams.validation_seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.hparams.validation_seed)
            logger.info(f"Using fixed seed {self.hparams.validation_seed} for validation generation.")

        # Set DDIM scheduler timesteps for inference
        self.inference_scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        # Encode text prompts to embeddings.
        # The _encode_text method should return `text_embeddings` in the appropriate
        # dtype (e.g., bfloat16 if QLoRA compute_dtype is bf16).
        # We do not need an explicit .to(self.torch_dtype) here, as autocast
        # in the denoising loop will handle type consistency if needed.
        encoder_hidden_states = self._encode_text(prompts) 
        
        # Determine initial latent shape based on image size and UNet config
        latents_shape = (
            len(prompts),
            self.unet.config.in_channels,
            self.hparams.image_size // 8,
            self.hparams.image_size // 8,
        )

        # Initialize random latents with the model's preferred compute dtype
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device=self.device,
            dtype=self.torch_dtype, # Use the model's compute dtype
        )
        
        # Scale latents for inference as per diffusers pipeline
        latents = latents * self.inference_scheduler.init_noise_sigma

        logger.info(f"Starting diffusion inference for {len(prompts)} samples.")
        # Diffusion loop for image generation
        # Use autocast here to ensure operations within UNet are performed at the correct precision.
        with torch.autocast(device_type=self.device.type, dtype=self.torch_dtype, enabled=self.hparams.mixed_precision != "fp32"):
            for t in tqdm(self.inference_scheduler.timesteps, desc="Generating Samples"):
                latent_model_input = self.inference_scheduler.scale_model_input(latents, t)
                # UNet inference. Its inputs should be correctly typed by autocast.
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states).sample
                latents = self.inference_scheduler.step(noise_pred, t, latents).prev_sample

        # Decode final latents to pixel values
        logger.info("Decoding latents to images.")
        return self._decode_vae(latents)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures the optimizer and learning rate scheduler for training.

        Uses AdamW or 8-bit AdamW if available, and a OneCycleLR scheduler.

        Returns:
            Dict[str, Any]: A dictionary containing the optimizer and LR scheduler configuration.

        Raises:
            ValueError: If no trainable parameters are found.
        """
        optimizer_cls = AdamW
        if self.hparams.use_8bit_adam:
            # Check if bitsandbytes is available and has AdamW8bit
            if bnb is not None and hasattr(bnb.optim, "AdamW8bit"):
                optimizer_cls = bnb.optim.AdamW8bit
                logger.info("Using 8-bit AdamW optimizer.")
            else:
                # Clarified message: 8-bit optimizer, not model quantization
                logger.warning(
                    "use_8bit_adam=True, but bitsandbytes.optim.AdamW8bit is not available. "
                    "Falling back to standard AdamW. "
                    "Ensure 'bitsandbytes' is installed for 8-bit optimizer support."
                )

        param_groups = []

        # Collect trainable UNet parameters
        unet_params = [p for p in self.unet.parameters() if p.requires_grad]
        if unet_params:
            param_groups.append({"params": unet_params, "lr": self.hparams.learning_rate})

        # Conditionally collect trainable Text Encoder parameters
        if self.hparams.train_text_encoder_lora:
            text_encoder_params = [p for p in self.text_encoder.parameters() if p.requires_grad]
            if text_encoder_params:
                param_groups.append({
                    "params": text_encoder_params,
                    "lr": self.hparams.learning_rate * self.hparams.text_encoder_lora_lr_scale
                })

        # Raise an error if no parameters are set to be trainable
        if not param_groups:
            raise ValueError("No trainable parameters found. Check LoRA configuration and `requires_grad` settings.")

        # Initialize the optimizer with collected parameter groups
        optimizer = optimizer_cls(
            param_groups,
            betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
            weight_decay=self.hparams.adam_weight_decay,
            eps=self.hparams.adam_epsilon,
        )
        logger.info(f"Optimizer '{optimizer_cls.__name__}' initialized.")


        # Calculate total_steps for OneCycleLR scheduler
        # It's more robust to rely on trainer properties directly or pass total_steps as hparam.
        # This logic handles cases where dataloader length might not be immediately available
        # or max_steps is explicitly set.
        total_steps = 0
        try:
            if self.trainer.max_steps and self.trainer.max_steps > 0:
                total_steps = self.trainer.max_steps
                logger.debug(f"Total steps set by trainer.max_steps: {total_steps}")
            else:
                # Ensure datamodule and train_dataloader are available
                if not hasattr(self.trainer, 'datamodule') or not hasattr(self.trainer.datamodule, 'train_dataloader'):
                    logger.warning(
                        "Datamodule or its train_dataloader is not available for calculating total_steps. "
                        "Cannot precisely estimate total_steps for LR scheduler."
                    )
                    # Fallback to a large number if total steps cannot be determined,
                    # so scheduler doesn't crash but might not be optimal.
                    total_steps = 1_000_000 
                else:
                    train_dataloader = self.trainer.datamodule.train_dataloader()
                    if hasattr(train_dataloader, '__len__') and len(train_dataloader) > 0:
                        total_steps = self.trainer.max_epochs * len(train_dataloader)
                        logger.debug(f"Estimated total steps from dataloader and max_epochs: {total_steps}")
                    else:
                        logger.warning("Train dataloader has no length. Cannot precisely estimate total_steps. Using large fallback.")
                        total_steps = 1_000_000 # Fallback

            if total_steps <= 0: # Defensive check
                logger.error("Calculated total_steps is non-positive. Setting to fallback value of 1,000,000.")
                total_steps = 1_000_000

        except Exception as e:
            logger.error(f"Error calculating total_steps for LR scheduler: {e}. Using fallback 1,000_000.")
            total_steps = 1_000_000 # Final robust fallback

        # Initialize OneCycleLR scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[pg.get("lr", self.hparams.learning_rate) for pg in optimizer.param_groups],
            total_steps=total_steps,
            # Common OneCycleLR parameters; adjust as needed
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1e4
        )
        logger.info(f"OneCycleLR scheduler initialized with total_steps={total_steps}.")


        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures the optimizer and learning rate scheduler for training.

        Uses AdamW or 8-bit AdamW if available, and a OneCycleLR scheduler.

        Returns:
            Dict[str, Any]: A dictionary containing the optimizer and LR scheduler configuration.

        Raises:
            ValueError: If no trainable parameters are found.
        """
        optimizer_cls = AdamW
        if self.hparams.use_8bit_adam:
            # Check if bitsandbytes is available and has AdamW8bit
            if bnb is not None and hasattr(bnb.optim, "AdamW8bit"):
                optimizer_cls = bnb.optim.AdamW8bit
                logger.info("Using 8-bit AdamW optimizer.")
            else:
                # Clarified message: 8-bit optimizer, not model quantization
                logger.warning(
                    "use_8bit_adam=True, but bitsandbytes.optim.AdamW8bit is not available. "
                    "Falling back to standard AdamW. "
                    "Ensure 'bitsandbytes' is installed for 8-bit optimizer support."
                )

        param_groups = []

        # Collect trainable UNet parameters
        unet_params = [p for p in self.unet.parameters() if p.requires_grad]
        if unet_params:
            param_groups.append({"params": unet_params, "lr": self.hparams.learning_rate})

        # Conditionally collect trainable Text Encoder parameters
        if self.hparams.train_text_encoder_lora:
            text_encoder_params = [p for p in self.text_encoder.parameters() if p.requires_grad]
            if text_encoder_params:
                param_groups.append({
                    "params": text_encoder_params,
                    "lr": self.hparams.learning_rate * self.hparams.text_encoder_lora_lr_scale
                })

        # Raise an error if no parameters are set to be trainable
        if not param_groups:
            raise ValueError("No trainable parameters found. Check LoRA configuration and `requires_grad` settings.")

        # Initialize the optimizer with collected parameter groups
        optimizer = optimizer_cls(
            param_groups,
            betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
            weight_decay=self.hparams.adam_weight_decay,
            eps=self.hparams.adam_epsilon,
        )
        logger.info(f"Optimizer '{optimizer_cls.__name__}' initialized.")

        total_steps = 0
        try:
            if self.trainer.max_steps and self.trainer.max_steps > 0:
                total_steps = self.trainer.max_steps
                logger.debug(f"Total steps set by trainer.max_steps: {total_steps}")
            else:
                # Ensure datamodule and train_dataloader are available
                if not hasattr(self.trainer, 'datamodule') or not hasattr(self.trainer.datamodule, 'train_dataloader'):
                    logger.warning(
                        "Datamodule or its train_dataloader is not available for calculating total_steps. "
                        "Cannot precisely estimate total_steps for LR scheduler."
                    )
                    # Fallback to a large number if total steps cannot be determined,
                    # so scheduler doesn't crash but might not be optimal.
                    total_steps = 1_000_000 
                else:
                    train_dataloader = self.trainer.datamodule.train_dataloader()
                    if hasattr(train_dataloader, '__len__') and len(train_dataloader) > 0:
                        total_steps = self.trainer.max_epochs * len(train_dataloader)
                        logger.debug(f"Estimated total steps from dataloader and max_epochs: {total_steps}")
                    else:
                        logger.warning("Train dataloader has no length. Cannot precisely estimate total_steps. Using large fallback.")
                        total_steps = 1_000_000 # Fallback

            if total_steps <= 0: # Defensive check
                logger.error("Calculated total_steps is non-positive. Setting to fallback value of 1,000,000.")
                total_steps = 1_000_000

        except Exception as e:
            logger.error(f"Error calculating total_steps for LR scheduler: {e}. Using fallback 1,000_000.")
            total_steps = 1_000_000 # Final robust fallback

        # Initialize OneCycleLR scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[pg.get("lr", self.hparams.learning_rate) for pg in optimizer.param_groups],
            total_steps=total_steps,
            # Common OneCycleLR parameters; adjust as needed
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1e4
        )
        logger.info(f"OneCycleLR scheduler initialized with total_steps={total_steps}.")


        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Saves the LoRA adapter weights in a diffusers-compatible format.

        This method saves the LoRA adapter weights (from UNet and Text Encoder)
        to a fixed directory, overwriting previous files to keep only the
        last checkpoint. This structure can be easily loaded by
        `diffusers.StableDiffusionPipeline` using its `load_lora_weights()` method.

        Args:
            checkpoint: The checkpoint dictionary provided by PyTorch Lightning.
        """
        if self.trainer and hasattr(self.trainer.checkpoint_callback, "dirpath"):
            # Determine the base directory for saving LoRA weights.
            base_save_dir = Path(self.trainer.checkpoint_callback.dirpath)
            
            # Use a fixed directory name to save only the last checkpoint's adapters.
            # This ensures that subsequent saves overwrite the previous ones.
            lora_output_dir = base_save_dir / "last_lora_checkpoint"
            lora_output_dir.mkdir(exist_ok=True, parents=True) # Ensure directory exists

            logger.info(f"Saving LoRA adapters to: {lora_output_dir}")

            # Save UNet LoRA adapter weights and config
            unet_save_path = lora_output_dir / "unet"
            unet_save_path.mkdir(exist_ok=True)
            if isinstance(self.unet, PeftModel):
                # PeftModel's save_pretrained saves adapter_model.safetensors and adapter_config.json
                self.unet.save_pretrained(str(unet_save_path), safe_serialization=True)
                logger.info(f"Saved UNet LoRA adapter weights and config to {unet_save_path}")
            else:
                logger.warning("UNet is not a PeftModel. Skipping UNet LoRA save.")

            # Save Text Encoder LoRA adapter weights and config if enabled
            if self.hparams.train_text_encoder_lora:
                text_encoder_save_path = lora_output_dir / "text_encoder"
                text_encoder_save_path.mkdir(exist_ok=True)
                if isinstance(self.text_encoder, PeftModel):
                    # PeftModel's save_pretrained saves adapter_model.safetensors and adapter_config.json
                    self.text_encoder.save_pretrained(str(text_encoder_save_path), safe_serialization=True)
                    logger.info(f"Saved Text Encoder LoRA adapter weights and config to {text_encoder_save_path}")
                else:
                    logger.warning("Text Encoder is not a PeftModel. Skipping Text Encoder LoRA save.")
            
            logger.info(f"Successfully saved LoRA weights to {lora_output_dir}")

        else:
            logger.warning("Could not determine checkpoint save path. Skipping separate LoRA save.")


    @classmethod
    def load_lora_weights_from_path(
        cls,
        lora_weights_dir: str,
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
        map_location: Optional[str] = None,
        # All hyperparameters required by LoraDiffusionModel's __init__ must be provided
        # so that the base model is correctly initialized and LoRA adapters can be attached.
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        unet_lora_target_modules: Tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0"),
        train_text_encoder_lora: bool = True,
        text_encoder_lora_target_modules: Tuple[str, ...] = ("q_proj", "v_proj"),
        text_encoder_lora_rank: Optional[int] = None,
        text_encoder_lora_alpha: Optional[int] = None,
        learning_rate: float = 1e-4, # Ensure all base hparams are here
        text_encoder_lora_lr_scale: float = 1.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        use_8bit_adam: bool = False,
        use_4bit_quantization: bool = False,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_compute_dtype: str = "bfloat16",
        mixed_precision: str = "bf16",
        gradient_checkpointing: bool = True,
        validation_metric: str = "LPIPS",
        clip_model_name_or_path: str = "openai/clip-vit-large-patch14",
        lpips_net_type: str = "vgg",
        fid_feature_size: int = 2048,
        num_validation_images: int = 4,
        val_num_inference_steps: int = 25,
        log_every_n_epochs: int = 1,
        image_size: int = 512,
        validation_seed: Optional[int] = 42,
        vae_name_or_path: Optional[str] = None 
    ) -> "LoraDiffusionModel":
        """
        Loads a LoraDiffusionModel by first initializing the base model (with empty LoRA adapters)
        and then loading the specific LoRA weights from the provided directory.

        Args:
            lora_weights_dir: Path to the directory containing the saved LoRA weights.
                              This directory is expected to contain 'unet/' and optionally
                              'text_encoder/' subdirectories with 'adapter_model.safetensors'
                              (or '.bin') and 'adapter_config.json' inside them.
            pretrained_model_name_or_path: The original base model identifier.
            map_location: Device to map the loaded weights to (e.g., 'cpu', 'cuda:0').
                          If None, defaults to current device.
            **kwargs: All hyperparameters required by LoraDiffusionModel's __init__ must be explicitly
                      passed here to ensure proper model re-initialization before loading adapters.

        Returns:
            An instance of LoraDiffusionModel with loaded LoRA weights.
        """
        logger.info(f"Loading LoRA weights from directory: {lora_weights_dir}")

        # Instantiate the model with its full configuration. This will load the base
        # models and apply *empty* LoRA adapters based on the provided hparams.
        model = cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            vae_name_or_path=vae_name_or_path, # Pass vae_name_or_path
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            unet_lora_target_modules=unet_lora_target_modules,
            train_text_encoder_lora=train_text_encoder_lora,
            text_encoder_lora_target_modules=text_encoder_lora_target_modules,
            text_encoder_lora_rank=text_encoder_lora_rank,
            text_encoder_lora_alpha=text_encoder_lora_alpha,
            learning_rate=learning_rate,
            text_encoder_lora_lr_scale=text_encoder_lora_lr_scale,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_weight_decay=adam_weight_decay,
            adam_epsilon=adam_epsilon,
            use_8bit_adam=use_8bit_adam,
            use_4bit_quantization=use_4bit_quantization,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            mixed_precision=mixed_precision,
            gradient_checkpointing=gradient_checkpointing,
            validation_metric=validation_metric,
            clip_model_name_or_path=clip_model_name_or_path,
            lpips_net_type=lpips_net_type,
            fid_feature_size=fid_feature_size,
            num_validation_images=num_validation_images,
            val_num_inference_steps=val_num_inference_steps,
            log_every_n_epochs=log_every_n_epochs,
            image_size=image_size,
            validation_seed=validation_seed,
        )
        
        # Ensure model is on the correct device for loading weights
        # Lightning's device property is typically set during setup.
        # Explicitly moving here if map_location is provided.
        if map_location:
            model.to(map_location)
        else:
            # If map_location is not specified, ensure model is on the default device
            # that Lightning would use (e.g., 'cuda:0' if GPU is available).
            # This is more robust than assuming model.device is already correct for external loading.
            if torch.cuda.is_available():
                model.to("cuda")


        try:
            # Load the LoRA adapter weights directly into the existing PeftModel instances.
            # PEFT's from_pretrained method correctly handles loading from a directory
            # that contains 'adapter_config.json' and 'adapter_model.safetensors' (or .bin).
            
            # Load UNet LoRA adapters
            unet_lora_path = Path(lora_weights_dir) / "unet"
            if unet_lora_path.is_dir():
                # The first argument is the base model (wrapped by PeftModel in this case)
                # The second argument is the path to the adapter weights directory
                model.unet = PeftModel.from_pretrained(model.unet, str(unet_lora_path))
                logger.info(f"Loaded UNet LoRA adapter weights from {unet_lora_path}")
            else:
                logger.warning(f"UNet LoRA directory not found at {unet_lora_path}. Skipping UNet LoRA load.")
            
            # Load Text Encoder LoRA adapters if enabled
            if model.hparams.train_text_encoder_lora:
                text_encoder_lora_path = Path(lora_weights_dir) / "text_encoder"
                if text_encoder_lora_path.is_dir():
                    model.text_encoder = PeftModel.from_pretrained(model.text_encoder, str(text_encoder_lora_path))
                    logger.info(f"Loaded Text Encoder LoRA adapter weights from {text_encoder_lora_path}")
                else:
                    logger.warning(f"Text Encoder LoRA directory not found at {text_encoder_lora_path}. Skipping Text Encoder LoRA load.")

            logger.info(f"Successfully loaded LoRA weights from {lora_weights_dir}.")
            
            # After loading, it's good practice to re-apply freezing and set modes
            # This ensures consistency whether it's for continued training or inference.
            model._freeze_models() # Re-freeze base models

            # Set modules to train mode if you intend to continue training.
            # If this model instance is purely for inference, then model.eval() would be appropriate.
            # For a training setup that loads previous weights, setting to train is usually the goal.
            model.unet.train()
            if model.hparams.train_text_encoder_lora:
                model.text_encoder.train()
            
        except Exception as e:
            logger.error(f"Error loading LoRA weights from {lora_weights_dir}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load LoRA weights for resuming training/inference: {e}")

        return model
