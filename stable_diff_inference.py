# inference.py
# -*- coding: utf-8 -*-
"""
Generates comparison images using a base Stable Diffusion model and a
fine-tuned LoRA version.

This script loads a base model and a LoRA checkpoint, then iterates through
a list of prompts provided via Hydra configuration. For each prompt, it
generates a specified number of image pairs (base vs. fine-tuned) using
consistent seeds for comparison.

It saves only the comparison images, which display the base and fine-tuned
results side-by-side with a white border at the top containing labels and
a white border at the bottom containing the prompt, seed, and optionally the LPIPS score.
Outputs are saved directly into the Hydra run directory.
"""

import os
import sys
import random
import logging
import textwrap
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union

import hydra
from hydra.core.hydra_config import HydraConfig

import numpy as np
import torch
from diffusers import (DDIMScheduler, DPMSolverMultistepScheduler,
                       StableDiffusionPipeline)
from omegaconf import DictConfig, ListConfig, OmegaConf
from hydra.utils import get_original_cwd
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

# --- Dependency Availability Checks ---

try:
    # Attempt to import the custom LoraDiffusionModel
    # Ensure 'src' is in the Python path or installed
    from src.models.stable_diffusion import LoraDiffusionModel
    _MODEL_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Could not import LoraDiffusionModel: {e}")
    print("Ensure 'src' directory is in PYTHONPATH or the package is installed.")
    _MODEL_AVAILABLE = False
    LoraDiffusionModel = None # Define as None if import fails

try:
    # Attempt to import LPIPS from torchmetrics
    from torchmetrics.image import \
        LearnedPerceptualImagePatchSimilarity as LPIPS
    _LPIPS_AVAILABLE = True
except ImportError:
    print("INFO: LPIPS metric unavailable. To calculate LPIPS score, install"
          " torchmetrics with image support: pip install torchmetrics[image]")
    _LPIPS_AVAILABLE = False
    LPIPS = None # Define as None if import fails

# Import DiffusionDataModule and DiffusionDataset
try:
    from src.data.stable_diffusion.datamodule import DiffusionDataModule
    from src.data.stable_diffusion.dataset import DiffusionDataset
    _DATAMODULE_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Could not import DiffusionDataModule or DiffusionDataset: {e}")
    print("Ensure 'src' directory is in PYTHONPATH or the package is installed.")
    _DATAMODULE_AVAILABLE = False
    DiffusionDataModule = None
    DiffusionDataset = None


# --- Global Logger ---
logger = logging.getLogger(__name__)


# --- Helper Functions ---

def get_font(
    font_path: str = "arial.ttf", size: int = 15
) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
    """
    Loads a TrueType font from the specified path or falls back to PIL's
    default bitmap font if the TrueType font is not found.

    Args:
        font_path: Path to the .ttf font file.
        size: Desired font size.

    Returns:
        A PIL font object (either FreeTypeFont or ImageFont).
    """
    try:
        font = ImageFont.truetype(font_path, size)
    except IOError:
        logger.warning(
            f"'{font_path}' not found. Falling back to PIL default font. "
            f"Text quality may vary. Install '{font_path}' for better results."
        )
        try:
            # Pillow >= 10 load_default does not accept size
            if hasattr(ImageFont, "load_default") and \
               'size' in ImageFont.load_default.__code__.co_varnames:
                font = ImageFont.load_default(size=size)
            else:
                font = ImageFont.load_default()
        except Exception as e:
            logger.error(f"Could not load default PIL font: {e}")
            # Provide a dummy font object to avoid crashes in drawing functions
            class DummyFont:
                def getbbox(self, *args, **kwargs): return (0, 0, 0, 0)
                def getlength(self, *args, **kwargs): return 0
            font = DummyFont()
    return font

def validate_config(cfg: DictConfig):
    """
    Validates the Hydra configuration object for essential parameters.

    Args:
        cfg: The DictConfig object loaded by Hydra.

    Raises:
        SystemExit: If critical configuration errors are found.
    """
    errors = []
    if not cfg.get("ckpt_path"):
        errors.append("'ckpt_path' (fine-tuned checkpoint) is missing.")
    if not cfg.get("prompts") or not OmegaConf.is_list(cfg.prompts) or \
       len(cfg.prompts) == 0:
        errors.append("'prompts' must be a non-empty list of strings.")
    if cfg.get("num_images_per_prompt", 1) <= 0:
        errors.append("'num_images_per_prompt' must be positive.")
    if cfg.generation.get("steps", 1) < 1:
        errors.append("'generation.steps' must be >= 1.")
    if cfg.generation.get("guidance_scale", 0.0) < 0:
        errors.append("'generation.guidance_scale' cannot be negative.")

    allowed_schedulers = ["ddim", "dpm"]
    scheduler_cfg = cfg.generation.get("scheduler_type", "dpm")
    if not scheduler_cfg or scheduler_cfg.lower() not in allowed_schedulers:
        errors.append(f"'generation.scheduler_type' must be one of "
                      f"{allowed_schedulers}.")

    allowed_devices = ["cuda", "cpu"]
    device_cfg = cfg.compute.get("device", "cuda")
    if not device_cfg or device_cfg.lower() not in allowed_devices:
        errors.append(f"'compute.device' must be one of {allowed_devices}.")

    allowed_dtypes = ["float16", "bfloat16", "float32", None]
    dtype_cfg = cfg.compute.get("dtype", None)
    if dtype_cfg not in allowed_dtypes:
        errors.append(f"'compute.dtype' must be one of {allowed_dtypes} "
                      f"or null.")

    if not _MODEL_AVAILABLE:
        errors.append("Critical: LoraDiffusionModel class not found. Check "
                      "src/models/lora_diffusion.py and import paths.")

    if cfg.get("calculate_lpips_with_val", False):
        if not _DATAMODULE_AVAILABLE:
            errors.append("Critical: 'calculate_lpips_with_val' is True, but DiffusionDataModule or DiffusionDataset not found.")
        if not cfg.get("val_data_path"):
            errors.append("Critical: 'val_data_path' must be specified in config for LPIPS with validation set.")
        if not cfg.get("val_class_name"):
            errors.append("Critical: 'val_class_name' must be specified in config for LPIPS with validation set.")
        if cfg.get("val_image_count", 0) <= 0:
            errors.append("Critical: 'val_image_count' must be positive for LPIPS with validation set.")
        if not _LPIPS_AVAILABLE:
            errors.append("Warning: 'calculate_lpips_with_val' is True, but LPIPS dependency not installed (torchmetrics[image]).")


    if errors:
        for e in errors:
            logger.error(f"Config error: {e}")
        logger.critical("Exiting due to configuration errors.")
        sys.exit(1)
    else:
        logger.info("Configuration validation passed.")


def create_comparison_image_with_border(
    img1: Image.Image,
    img2: Image.Image,
    prompt: str,
    seed: int,
    label1: str = "Base",
    label2: str = "Fine-tuned",
    # Removed lpips_score parameter
    lpips_with_val_score: Optional[float] = None,
    font_size: int = 14,
    top_border_height: int = 40,
    bottom_border_height_initial: int = 60,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
    spacing: int = 10
) -> Image.Image:
    """
    Creates a side-by-side comparison image using PIL with annotation
    borders at the top and bottom.

    Args:
        img1: The first PIL Image (e.g., base model output).
        img2: The second PIL Image (e.g., fine-tuned model output).
        prompt: The text prompt used for generation.
        seed: The seed used for generation.
        label1: Label for the first image (e.g., "Base Model").
        label2: Label for the second image (e.g., "Fine-tuned Model").
        lpips_with_val_score: Optional LPIPS score with validation image.
        font_size: Font size for annotations.
        top_border_height: Height of the top border for labels.
        bottom_border_height_initial: Fallback height if dynamic calculation fails for bottom border.
        bg_color: Background color of the canvas and border.
        text_color: Color for the annotation text.
        spacing: Horizontal spacing between the two images.

    Returns:
        A new PIL Image containing the comparison with the annotation borders.
    """
    bottom_border_height = bottom_border_height_initial

    try:
        # --- Font Loading ---
        font = get_font(size=font_size)
        label_font = get_font(size=font_size + 2)

        # Ensure valid image dimensions before proceeding
        if img1.width <= 0 or img1.height <= 0 or \
           img2.width <= 0 or img2.height <= 0:
            raise ValueError("Input images must have positive dimensions.")

        # Resize images to have the same height for consistent layout
        max_h = max(img1.height, img2.height)
        if img1.height != max_h:
            ratio = max_h / img1.height
            new_w1 = int(img1.width * ratio)
            if new_w1 > 0:
                img1 = img1.resize((new_w1, max_h), Image.Resampling.LANCZOS)
            else: raise ValueError("Calculated width for img1 is not positive.")
        if img2.height != max_h:
            ratio = max_h / img2.height
            new_w2 = int(img2.width * ratio)
            if new_w2 > 0:
                img2 = img2.resize((new_w2, max_h), Image.Resampling.LANCZOS)
            else: raise ValueError("Calculated width for img2 is not positive.")

        # Use width after potential resize for text wrapping calculation
        img_width = img1.width

        # --- Text Preparation for bottom border ---
        approx_chars_per_pixel = 0.6
        wrap_width_multiplier = 1.8
        min_wrap_width = 20

        if font_size > 0 and approx_chars_per_pixel > 0:
            chars_per_img_width = img_width / (font_size * approx_chars_per_pixel)
            wrap_width = int(chars_per_img_width * wrap_width_multiplier)
            wrap_width = max(wrap_width, min_wrap_width)
        else:
            wrap_width = 80

        wrapped_prompt = textwrap.fill(f"Prompt: {prompt}", width=wrap_width)
        seed_text = f"Seed: {seed}"
        # Removed lpips_text
        lpips_val_text = f"LPIPS (Fine-tuned vs Val): {lpips_with_val_score:.4f}" if lpips_with_val_score is not None \
                                 else "LPIPS (Fine-tuned vs Val): N/A"

        # Only include lpips_val_text if it's available, otherwise omit it.
        # This will prevent "LPIPS: N/A" from showing if the calculation is off
        annotation_lines = [seed_text]
        if lpips_with_val_score is not None:
             annotation_lines.insert(0, lpips_val_text)
        annotation_lines.extend(wrapped_prompt.split('\n'))


        # --- Calculate Layout Dynamically for bottom border ---
        max_line_height = 0
        total_text_height = 0
        line_spacing_factor = 1.3
        vertical_padding = 20

        for line in annotation_lines:
            try:
                line_bbox = font.getbbox(line)
                if line_bbox and len(line_bbox) == 4:
                    line_height = line_bbox[3] - line_bbox[1]
                else:
                    line_height = font_size
                max_line_height = max(max_line_height, line_height)
                total_text_height += line_height * line_spacing_factor
            except (AttributeError, TypeError):
                max_line_height = max(max_line_height, font_size)
                total_text_height += font_size * line_spacing_factor

        bottom_border_height = int(total_text_height) + vertical_padding

        # Total dimensions for the final image
        total_width = img1.width + spacing + img2.width
        total_height = max_h + top_border_height + bottom_border_height

        # --- Create Canvas and Draw ---
        canvas = Image.new('RGB', (total_width, total_height), bg_color)
        draw = ImageDraw.Draw(canvas)

        # Paste images onto the canvas, offset by top_border_height
        canvas.paste(img1, (0, top_border_height))
        canvas.paste(img2, (img1.width + spacing, top_border_height))

        # Draw labels in the top border
        label1_bbox = label_font.getbbox(label1)
        label2_bbox = label_font.getbbox(label2)

        label1_x = max((img1.width - (label1_bbox[2] - label1_bbox[0])) // 2, 5)
        label2_x = img1.width + spacing + max((img2.width - (label2_bbox[2] - label2_bbox[0])) // 2, 5)
        label_y = (top_border_height - (label1_bbox[3] - label1_bbox[1])) // 2

        draw.text((label1_x, label_y), label1, font=label_font, fill=text_color)
        draw.text((label2_x, label_y), label2, font=label_font, fill=text_color)

        # Draw annotation text in the bottom border
        current_y = max_h + top_border_height + (vertical_padding // 2)
        horizontal_margin = 10

        for line in annotation_lines:
            try:
                line_bbox = font.getbbox(line)
                if line_bbox and len(line_bbox) == 4:
                    line_width = line_bbox[2] - line_bbox[0]
                    line_height = line_bbox[3] - line_bbox[1]
                else:
                    line_width = font.getlength(line) if hasattr(font, 'getlength') else len(line)*font_size*0.6
                    line_height = font_size

                # Center text horizontally within the canvas width
                line_x = max((total_width - line_width) // 2, horizontal_margin)
                draw.text((line_x, current_y), line, font=font, fill=text_color)
                current_y += line_height * line_spacing_factor
            except (AttributeError, TypeError):
                # Fallback drawing if font methods fail
                draw.text((horizontal_margin, current_y), line, fill=text_color)
                current_y += font_size * line_spacing_factor

        return canvas

    except Exception as e:
        logger.exception(f"Failed to create comparison image with border: {e}")
        # Return a placeholder grey image on failure
        fallback_width = 512 + spacing + 512
        fallback_img_h = 512
        fallback_total_h = fallback_img_h + top_border_height + (bottom_border_height if bottom_border_height > 0 else 60)
        return Image.new(
            'RGB',
            (fallback_width, fallback_total_h),
            color='grey'
            )


# --- Main Inference Function ---
@hydra.main(config_path="config", config_name="infer", version_base=None)
def main(cfg: DictConfig):
    """
    Main function to run the inference process based on Hydra configuration.

    Loads models, generates image pairs for each prompt, calculates LPIPS
    (if enabled), and saves comparison images with annotations.

    Args:
        cfg: The DictConfig object loaded by Hydra.
    """
    logger.info("--- Starting Inference ---")
    logger.info(f"Full Configuration:\n{OmegaConf.to_yaml(cfg)}")
    validate_config(cfg)

    try:
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        logger.info(f"Hydra Runtime Output Directory: {output_dir}")
    except Exception as e:
        logger.warning(f"Could not get output dir from HydraConfig ({e}), falling back to CWD.")
        output_dir = Path(os.getcwd())
        logger.info(f"Output Directory (fallback CWD): {output_dir}")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create or access output directory {output_dir}: {e}")
        sys.exit(1)

    # --- Compute Setup ---
    device = cfg.compute.device.lower()
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA specified but unavailable, falling back to CPU.")
        device = "cpu"
    logger.info(f"Using device: {device}")

    compute_dtype_str = cfg.compute.get("dtype", None)
    if compute_dtype_str is None:
        compute_dtype_str = "float16" if device == "cuda" else "float32"
        logger.info(f"Compute dtype not specified, defaulting to {compute_dtype_str}")

    if device == "cpu" and compute_dtype_str != "float32":
        logger.warning("CPU selected, forcing compute dtype to float32.")
        compute_dtype_str = "float32"

    try:
        inference_dtype = getattr(torch, compute_dtype_str)
    except AttributeError:
        logger.warning(f"Invalid compute dtype '{compute_dtype_str}', "
                        f"defaulting to float32.")
        inference_dtype = torch.float32
    logger.info(f"Using inference dtype: {inference_dtype}")

    # --- Load Base Model Pipeline ---
    base_model_name = cfg.get("base_model_name",
                              "runwayml/stable-diffusion-v1-5")
    try:
        logger.info(f"Loading base pipeline: {base_model_name}")
        base_pipeline = StableDiffusionPipeline.from_pretrained(
            base_model_name, torch_dtype=inference_dtype
        )
        base_pipeline.to(device)
        if cfg.get("disable_safety_checker", True):
            base_pipeline.safety_checker = None
            logger.info("Base model safety checker disabled.")
        logger.info("Base pipeline loaded successfully.")
    except Exception as e:
        logger.exception(f"Failed to load base pipeline '{base_model_name}': {e}")
        sys.exit(1)

    # --- Load Fine-Tuned LoRA Model ---
    if not _MODEL_AVAILABLE:
        logger.critical("LoraDiffusionModel class unavailable. Cannot load fine-tuned model.")
        sys.exit(1)
    try:
        ckpt_path_raw = cfg.ckpt_path
        # Resolve relative path from original CWD where the script was invoked
        if not os.path.isabs(ckpt_path_raw):
            ckpt_path = Path(get_original_cwd()) / ckpt_path_raw
        else:
            ckpt_path = Path(ckpt_path_raw)

        logger.info(f"Attempting to load LoRA model from: {ckpt_path}")
        if not ckpt_path.exists():
            logger.error(f"Checkpoint not found at resolved path: {ckpt_path}")
            sys.exit(1)

        # Load Lightning Module checkpoint
        pl_model = LoraDiffusionModel.load_from_checkpoint(
            str(ckpt_path), map_location=device, strict=False
        )
        pl_model.eval()
        pl_model.to(device)
        logger.info("LoRA model loaded successfully from checkpoint.")
    except Exception as e:
        logger.exception(f"Failed to load LoRA model from checkpoint '{cfg.ckpt_path}': {e}")
        sys.exit(1)

    # --- Create Fine-Tuned Pipeline ---
    try:
        logger.info("Creating fine-tuned pipeline...")
        # Ensure VAE precision matches compute type if not float16
        vae_dtype = torch.float32 if inference_dtype == torch.float16 else inference_dtype

        # Extract components from the loaded Lightning module
        vae = pl_model.vae.to(device, dtype=vae_dtype)
        text_encoder = pl_model.text_encoder.to(device, dtype=inference_dtype)
        unet = pl_model.unet.to(device, dtype=inference_dtype)
        tokenizer = pl_model.tokenizer # Already loaded in pl_model

        # Setup Scheduler
        scheduler_type = cfg.generation.scheduler_type.lower()
        scheduler_class = DDIMScheduler if scheduler_type == "ddim" else DPMSolverMultistepScheduler

        # Use the same base model path for scheduler config as was used for training
        original_model_path = pl_model.hparams.get("pretrained_model_name_or_path", base_model_name)
        scheduler = scheduler_class.from_pretrained(
            original_model_path, subfolder="scheduler"
        )
        logger.info(f"Using scheduler {scheduler.__class__.__name__} configured from '{original_model_path}'.")

        # Assemble the fine-tuned pipeline
        fine_tuned_pipeline = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None, # Safety checker usually disabled for LoRA
            feature_extractor=None,
        )
        logger.info("Fine-tuned pipeline created successfully.")
    except Exception as e:
        logger.exception(f"Failed to create fine-tuned pipeline: {e}")
        sys.exit(1)

    # --- Setup LPIPS Metric ---
    lpips_metric = None
    lpips_transform = None
    # Only initialize LPIPS if calculate_lpips_with_val is True
    if cfg.get("calculate_lpips_with_val", False):
        if _LPIPS_AVAILABLE:
            logger.info("Setting up LPIPS calculation...")
            try:
                lpips_net = cfg.metrics.get('lpips_net_type', 'alex')
                logger.info(f"Using LPIPS network type: {lpips_net}")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    lpips_metric = LPIPS(net_type=lpips_net, normalize=False).to(device)
                lpips_transform = transforms.Compose([transforms.ToTensor()])
                logger.info("LPIPS metric and transform ready.")
            except Exception as e:
                logger.exception(f"Error initializing LPIPS metric: {e}")
                lpips_metric = None # Disable calculation on error
        else:
            logger.warning("LPIPS calculation requested but torchmetrics[image] not available.")

    # --- Load Validation Data for LPIPS (if enabled) ---
    val_dataset = None
    val_images_tensors: List[torch.Tensor] = []
    if cfg.get("calculate_lpips_with_val", False) and lpips_metric and _DATAMODULE_AVAILABLE:
        try:
            logger.info(f"Loading validation data from: {cfg.val_data_path}")
            # Instantiate DiffusionDataset directly for specific validation class images
            val_dataset = DiffusionDataset(
                data_path=cfg.val_data_path,
                transform=pl_model.val_transform,
                coco_image_subdir=cfg.get("coco_image_subdir", "images")
            )
            logger.info(f"Validation dataset instantiated with {len(val_dataset)} total samples.")

            target_class = cfg.val_class_name
            num_val_images = cfg.val_image_count

            logger.info(f"Sampling {num_val_images} random images for class '{target_class}' from validation set...")
            # get_random_images_for_class returns list of Tensors [C, H, W] in range [-1, 1]
            val_images_tensors = val_dataset.get_random_images_for_class(
                class_name=target_class,
                count=num_val_images,
                random_seed=cfg.get("seed", 42)
            )
            logger.info(f"Loaded {len(val_images_tensors)} validation images for LPIPS comparison.")

            if not val_images_tensors:
                logger.warning(f"No validation images found for class '{target_class}' or could not be loaded. LPIPS with validation will be skipped.")
                cfg.calculate_lpips_with_val = False # Disable if no images
        except Exception as e:
            logger.exception(f"Failed to load validation dataset for LPIPS: {e}")
            cfg.calculate_lpips_with_val = False # Disable on error

    # --- Setup Seed Generator ---
    master_seed = cfg.get("seed")
    generator = torch.Generator(device=device)

    if master_seed is not None and master_seed == -1:
        master_seed = random.randint(0, 2**32 - 1)
        logger.info(f"Using master random seed for the run: {master_seed}")
    elif master_seed is not None:
        logger.info(f"Using fixed master seed for the run: {master_seed}")
    else:
        master_seed = torch.initial_seed()
        logger.info(f"Seed not provided. Using default master seed: {master_seed}")
    generator.manual_seed(master_seed)

    # --- Generation Loop ---
    prompts: List[str] = cfg.prompts
    num_images_per_prompt: int = cfg.num_images_per_prompt
    neg_prompt: Optional[str] = cfg.get("negative_prompt")
    steps: int = cfg.generation.steps
    guidance: float = cfg.generation.guidance_scale
    save_comparison: bool = cfg.get("save_comparison_image", True)
    output_filename_prefix: str = cfg.get("output_filename", "output")

    # Removed all_lpips_scores as it's no longer used
    all_lpips_val_scores = []

    for p_idx, prompt in enumerate(prompts):
        logger.info(f"--- Processing Prompt {p_idx+1}/{len(prompts)}: '{prompt}' ---")
        logger.info(f"Generating {num_images_per_prompt} comparison image(s).")
        
        # Removed prompt_lpips_scores
        prompt_lpips_val_scores = []

        for i in range(num_images_per_prompt):
            current_seed = master_seed + p_idx * num_images_per_prompt + i
            logger.info(f"Generating image pair {i+1}/{num_images_per_prompt} "
                                f"for prompt {p_idx+1} with seed {current_seed}...")

            try:
                with torch.inference_mode(), \
                     torch.autocast(device_type=device.split(":")[0],
                                    dtype=inference_dtype,
                                    enabled=(inference_dtype != torch.float32)):

                    # Generate Base Image
                    logger.debug("Generating with base pipeline...")
                    generator.manual_seed(current_seed)
                    base_result = base_pipeline(
                        prompt=prompt,
                        negative_prompt=neg_prompt,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        generator=generator,
                    )
                    base_img = base_result.images[0]
                    logger.debug("Base image generated.")

                    # Generate Tuned Image
                    logger.debug("Generating with fine-tuned pipeline...")
                    generator.manual_seed(current_seed)
                    tuned_result = fine_tuned_pipeline(
                        prompt=prompt,
                        negative_prompt=neg_prompt,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        generator=generator,
                    )
                    tuned_img = tuned_result.images[0]
                    logger.debug("Fine-tuned image generated.")

                # Removed LPIPS Calculation (Base vs Fine-tuned)
                current_lpips = None # Ensure it remains None

                # --- LPIPS Calculation (Fine-tuned vs Validation) ---
                current_lpips_with_val = None
                if lpips_metric and lpips_transform and cfg.get("calculate_lpips_with_val", False) and val_images_tensors:
                    try:
                        # Convert generated image to tensor [0, 1]
                        tuned_tensor_for_lpips = lpips_transform(tuned_img).unsqueeze(0).to(device)

                        # Convert validation images from [-1, 1] (DiffusionDataset output) to [0, 1]
                        val_tensors_normalized = [
                            (img_t.float().to(device) + 1) / 2
                            for img_t in val_images_tensors
                        ]

                        # Calculate LPIPS between the tuned image and EACH sampled validation image
                        min_lpips_for_tuned_img = float('inf')
                        for val_t in val_tensors_normalized:
                            lpips_val_tensor = lpips_metric(tuned_tensor_for_lpips, val_t.unsqueeze(0))
                            if not torch.isnan(lpips_val_tensor) and not torch.isinf(lpips_val_tensor):
                                min_lpips_for_tuned_img = min(min_lpips_for_tuned_img, lpips_val_tensor.item())

                        if min_lpips_for_tuned_img != float('inf'):
                            current_lpips_with_val = min_lpips_for_tuned_img
                            logger.debug(f"Min LPIPS (Fine-tuned vs Val) score for pair {i+1}: {current_lpips_with_val:.4f}")
                        else:
                            logger.warning(f"LPIPS (Fine-tuned vs Val) calculation for pair {i+1} resulted in NaN/Inf or no valid comparisons.")
                    except Exception as e:
                        logger.error(f"Failed LPIPS (Fine-tuned vs Val) calculation for pair {i+1}: {e}", exc_info=True)
                prompt_lpips_val_scores.append(current_lpips_with_val)


                # --- Save Comparison Image ---
                if save_comparison:
                    prompt_prefix = "".join(
                        c if c.isalnum() else "_" for c in prompt[:30]
                    )
                    base_name = f"{prompt_prefix}_p{p_idx}_i{i}"
                    comp_filename = f"{output_filename_prefix}_{base_name}_comparison.png"
                    comp_path = output_dir / comp_filename

                    try:
                        logger.debug(f"Creating comparison image: {comp_path}")
                        comparison_image = create_comparison_image_with_border(
                            img1=base_img,
                            img2=tuned_img,
                            prompt=prompt,
                            seed=current_seed,
                            # Removed lpips_score=current_lpips
                            lpips_with_val_score=current_lpips_with_val,
                            font_size=cfg.get("annotation_font_size", 14)
                        )
                        comparison_image.save(comp_path)
                        # lpips_str = f"{current_lpips:.4f}" if current_lpips is not None else "N/A" # Removed
                        lpips_val_str = f"{current_lpips_with_val:.4f}" if current_lpips_with_val is not None else "N/A"
                        logger.info(f"Saved comparison image (LPIPS_Val: {lpips_val_str}): {comp_path}") # Modified log message
                    except Exception as e:
                        logger.error(f"Failed to create/save comparison image "
                                        f"'{comp_path}': {e}", exc_info=True)

            except Exception as e:
                logger.exception(f"Error during generation/saving for pair "
                                    f"{i+1} of prompt {p_idx+1}: {e}")
                continue

        # Removed all_lpips_scores.extend(prompt_lpips_scores)
        all_lpips_val_scores.extend(prompt_lpips_val_scores)

    # --- Final LPIPS Summary ---
    # Removed if cfg.get("calculate_lpips", False) block

    if cfg.get("calculate_lpips_with_val", False):
        valid_lpips_val_scores = [s for s in all_lpips_val_scores if s is not None]
        if valid_lpips_val_scores:
            avg_lpips_val = sum(valid_lpips_val_scores) / len(valid_lpips_val_scores)
            lpips_net_type = cfg.metrics.get('lpips_net_type', 'alex')
            logger.info(f"--- Overall Average LPIPS (Fine-tuned vs Validation) ({lpips_net_type}) over "
                                f"{len(valid_lpips_val_scores)} valid pairs: {avg_lpips_val:.4f} ---")
            try:
                score_filename = f"{output_filename_prefix}_lpips_tuned_val_AVG.txt"
                score_path = output_dir / score_filename
                with open(score_path, "w") as f:
                    f.write(f"{avg_lpips_val:.4f}\n")
                logger.info(f"Overall average LPIPS (Fine-tuned vs Validation) score saved to: {score_path}")
            except Exception as e:
                logger.error(f"Failed to save average LPIPS (Fine-tuned vs Validation) score: {e}")
        else:
            logger.warning("LPIPS (Fine-tuned vs Validation) calculation enabled, but no valid scores "
                               "were recorded to calculate an average.")

    logger.info("--- Inference Finished ---")


# --- Script Entry Point ---
if __name__ == "__main__":
    # Hydra decorator handles configuration loading and execution
    main()