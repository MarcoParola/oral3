# injection_visualizer.py
"""
A Hydra-configurable script to visualize the effect of conditional image injection
during the diffusion sampling process of a fine-tuned Stable Diffusion model.

This version uses a configurable seed for reproducible noise generation and
guidance image selection. It saves both a summary grid plot for each example
and the individual final images for each injection strength (lambda).

The generation process leverages Classifier-Free Guidance (CFG), a technique to
enhance prompt adherence. At each step, it calculates two noise predictions: one
conditioned on the text prompt and one unconditional (from an empty prompt).
The final prediction is then steered away from the unconditional estimate towards
the conditional one, with the strength of this guidance controlled by a scale
factor. This allows for robust control over the generated image's content.
"""

import logging
import random
from pathlib import Path
from typing import List

import torch
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from torchvision.utils import make_grid
from tqdm import tqdm

# --- Local Application Imports ---
try:
    from src.models.stable_diffusion import LoraDiffusionModel
    from src.data.stable_diffusion.dataset import DiffusionDataset
except ImportError as e:
    print("Error: Could not import local modules from the 'src' directory.")
    print("Please ensure you are running this script from the project's root folder.")
    print(f"Details: {e}")
    exit()

log = logging.getLogger(__name__)

def generate_with_injection(
    model: LoraDiffusionModel,
    prompt: str,
    guidance_latent: torch.Tensor,
    injection_step: int,
    cfg: DictConfig,
) -> List[torch.Tensor]:
    """Performs DDIM sampling with latent injection. Noise is now deterministic."""
    device = cfg.compute.device
    model.inference_scheduler.set_timesteps(cfg.generation.num_inference_steps, device=device)

    # --- Classifier-Free Guidance Setup ---
    uncond_embeddings = model._encode_text([""])
    text_embeddings = model._encode_text([prompt])
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    # --- Initial Latent Generation ---
    latents_shape = (1, model.unet.config.in_channels, model.hparams.image_size // 8, model.hparams.image_size // 8)
    latents = torch.randn(latents_shape, device=device, dtype=model.torch_dtype)
    latents = latents * model.inference_scheduler.init_noise_sigma

    captured_images = []
    # --- Denoising Loop ---
    desc = f"Denoising (injection at step {injection_step})"
    for i, t in enumerate(tqdm(model.inference_scheduler.timesteps, desc=desc)):
        # --- CORE INJECTION LOGIC ---
        if i == injection_step:
            noise = torch.randn_like(guidance_latent)
            latents = model.noise_scheduler.add_noise(guidance_latent, noise, t.unsqueeze(0)).to(model.torch_dtype)

        # --- Standard DDIM Step ---
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = model.inference_scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad(), torch.autocast(device_type=device, dtype=model.torch_dtype):
            noise_pred = model.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg.generation.guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = model.inference_scheduler.step(noise_pred, t, latents).prev_sample
        
        # --- Snapshot Capturing ---
        if (i + 1) % 10 == 0:
            with torch.no_grad():
                captured_images.append(model._decode_vae(latents.clone()).cpu())
                
    return captured_images

def save_individual_image(image_tensor: torch.Tensor, output_path: Path):
    """Saves a single image tensor to a file after normalization."""
    image_tensor_norm = (image_tensor.squeeze() / 2.0) + 0.5
    pil_image = Image.fromarray(
        image_tensor_norm.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    pil_image.save(output_path)
    # log.info(f"Saved individual image to: {output_path}") # Optional: uncomment for verbose logging

def save_grid_plot(
    snapshots: List[torch.Tensor],
    guidance_image: torch.Tensor,
    class_name: str,
    output_dir: Path,
    cfg: DictConfig,
    example_index: int
):
    """Creates and saves an annotated plot for a specific example."""
    if not snapshots:
        log.warning(f"No snapshots provided for class '{class_name}'. Skipping plot.")
        return

    # --- Tensor to Image Conversion ---
    all_snapshots_tensor = (torch.cat(snapshots) / 2.0) + 0.5
    grid_tensor = make_grid(all_snapshots_tensor, nrow=10, normalize=False)
    guidance_image_norm = (guidance_image / 2.0) + 0.5

    grid_pil = Image.fromarray(grid_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy())
    guidance_pil = Image.fromarray(guidance_image_norm.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy())
    
    # --- Plotting with Matplotlib ---
    fig = plt.figure(figsize=(16, 18))
    title = f"Conditional Injection Analysis for Class: '{class_name.upper()}' (Example {example_index})"
    fig.suptitle(title, fontsize=20, weight='bold')

    gs = gridspec.GridSpec(4, 1, figure=fig)
    
    ax_guidance = fig.add_subplot(gs[0, 0])
    ax_guidance.imshow(guidance_pil)
    ax_guidance.set_title("Original Guidance Image", fontsize=14)
    ax_guidance.axis('off')

    ax_grid = fig.add_subplot(gs[1:, 0])
    ax_grid.imshow(grid_pil)
    ax_grid.set_title("Generated Images Grid", fontsize=14, pad=20)
    
    # --- Annotate Axes ---
    ax_grid.set_xlabel("Diffusion Step", fontsize=12)
    x_ticks = [grid_pil.width * (i + 0.5) / 10 for i in range(10)]
    x_labels = [str((i + 1) * 10) for i in range(10)]
    ax_grid.set_xticks(x_ticks)
    ax_grid.set_xticklabels(x_labels)

    ax_grid.set_ylabel("Injection Lambda (λ)", fontsize=12)
    y_ticks = [grid_pil.height * (i + 0.5) / 10 for i in range(10)]
    y_labels = [f"{lam:.1f}" for lam in cfg.generation.lambda_values]
    ax_grid.set_yticks(y_ticks)
    ax_grid.set_yticklabels(y_labels)
    ax_grid.grid(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the grid plot in the main output directory
    save_path_annotated = output_dir / f"annotated_grid_{class_name}_example_{example_index}.png"
    plt.savefig(save_path_annotated)
    plt.close(fig)
    log.info(f"Annotated grid plot saved to: {save_path_annotated}")


@hydra.main(config_path="config", config_name="injection_vis", version_base=None)
def run_visualization(cfg: DictConfig):
    """Main execution function managed by Hydra."""
    # --- Reproducibility Setup ---
    if cfg.experiment.seed is not None:
        seed = cfg.experiment.seed
        log.info(f"Setting global seed to: {seed}")
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log.info(f"Configuration loaded. Outputs will be saved to: {output_dir}")
    log.info(f"Using device: {cfg.compute.device}")
    log.info(f"Active dataset: {cfg.active_dataset}")
    log.info(f"Prompt variation mode: {cfg.prompt_variation}")

    # 1. Load the fine-tuned model
    log.info(f"Loading model from: {cfg.paths.checkpoint}")
    try:
        device = cfg.compute.device
        model = LoraDiffusionModel.load_from_checkpoint(cfg.paths.checkpoint, map_location=device, strict=False)
        model.to(device).eval()
        log.info("Model loaded successfully.")
    except Exception as e:
        log.error(f"Failed to load model checkpoint: {e}", exc_info=True)
        return

    # 2. Load the guidance dataset
    log.info(f"Loading guidance data from: {cfg.paths.guidance_data}")
    try:
        dataset = DiffusionDataset(data_path=cfg.paths.guidance_data, coco_image_subdir=cfg.paths.coco_image_subdir)
        if len(dataset) == 0:
            log.error("Guidance dataset is empty. Please check the path.")
            return
    except Exception as e:
        log.error(f"Failed to load dataset: {e}", exc_info=True)
        return

    # 3. Iterate through each class in the dataset
    class_names = dataset.get_class_names()
    log.info(f"Found {len(class_names)} classes to process: {class_names}")

    for class_name in class_names:
        log.info(f"\n{'='*25} Processing Class: {class_name.upper()} {'='*25}")
        
        class_items = dataset.get_items_for_class(class_name)
        if not class_items:
            log.warning(f"No items found for class '{class_name}'. Skipping.")
            continue
        
        num_examples = min(5, len(class_items))
        if num_examples < 5:
            log.warning(f"Found only {num_examples} items for '{class_name}'. Will process {num_examples} examples.")
        
        chosen_items = random.sample(class_items, num_examples)

        for j, chosen_item in enumerate(chosen_items):
            example_index = j + 1
            log.info(f"\n--- Processing Example {example_index}/{num_examples} for class '{class_name}' ---")

            # Create dedicated output directory for this specific example
            example_output_dir = output_dir / class_name / str(example_index)
            example_output_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"Saving individual images to: {example_output_dir}")

            sample = dataset[chosen_item[2]]
            if sample is None:
                log.error(f"Failed to load sample for class '{class_name}', example {example_index}. Skipping.")
                continue

            prompt = sample["text"]
            guidance_image_tensor = sample["pixel_values"].unsqueeze(0).to(device)
            log.info(f"Selected guidance image for '{class_name}' with prompt: '{prompt}'")

            with torch.no_grad():
                guidance_latent = model._encode_vae(guidance_image_tensor)
            
            all_snapshots_for_grid = []
            # --- Loop over injection strengths (lambda) ---
            for i, lambda_val in enumerate(cfg.generation.lambda_values):
                injection_step = int(lambda_val * cfg.generation.num_inference_steps)
                log.info(f"--- Running generation for Lambda = {lambda_val:.1f} (Injection step: {injection_step}) ---")
                
                snapshots_for_lambda = generate_with_injection(
                    model=model, prompt=prompt, guidance_latent=guidance_latent,
                    injection_step=injection_step, cfg=cfg
                )
                
                if not snapshots_for_lambda:
                    log.warning("Generation returned no snapshots. Skipping save for this lambda.")
                    continue
                
                all_snapshots_for_grid.extend(snapshots_for_lambda)
                
                # --- Save the final generated image for this lambda value ---
                final_image_tensor = snapshots_for_lambda[-1]
                filename = f"{i}_{injection_step}_{lambda_val:.1f}.png"
                individual_image_path = example_output_dir / filename
                save_individual_image(final_image_tensor, individual_image_path)
            
            log.info(f"All lambda values processed for example {example_index}.")

            # --- Save the grid plot for the current example ---
            save_grid_plot(
                snapshots=all_snapshots_for_grid,
                guidance_image=sample["pixel_values"],
                class_name=class_name,
                output_dir=output_dir, # Base output dir for grids
                cfg=cfg,
                example_index=example_index
            )

    log.info("\n--- All classes and examples processed. Visualization complete. ---")

if __name__ == "__main__":
    run_visualization()