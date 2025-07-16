"""
Offline data augmentation script for creating augmented image datasets.

This script implements a 'DEFINE -> SYNTHESIZE -> MATERIALIZE' workflow:
1.  DEFINE: Calculates the number of images to generate per class based on a
    source dataset and a specified percentage.
2.  SYNTHESIZE: Generates new images using a selected engine (e.g., LoRA,
    StyleGAN, classic transforms).
3.  MATERIALIZE: Copies the original dataset and saves the new images and
    metadata to create a final, augmented dataset.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

# --- Standard Library Imports ---
import sys
import json
import shutil
import random
import logging
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Callable, Optional, Tuple

# --- Third-Party Imports ---
import hydra
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from tqdm import tqdm

# --- Local Application Imports ---
# Attempt to import optional LoRA dependencies.
try:
    from src.models.stable_diffusion import LoraDiffusionModel
    _LORA_MODEL_AVAILABLE = True
except ImportError:
    print("WARNING: Could not import LoraDiffusionModel. The 'lora_generate' engine will be unavailable.")
    LoraDiffusionModel = None
    _LORA_MODEL_AVAILABLE = False


# =============================================================================
# 2. SCRIPT SETUP
# =============================================================================

# --- Logger Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Global Cache for Generative Pipelines ---
# Caching loaded models avoids reloading them for each class, speeding up the process.
_LORA_PIPELINE_CACHE: Dict[str, StableDiffusionPipeline] = {}


# =============================================================================
# 3. WORKFLOW STAGE 1: STRATEGY DEFINITION
# =============================================================================

def define_augmentation_strategy(
    source_train_path: Path,
    generation_percentage: float
) -> Tuple[Optional[Dict[str, int]], Optional[str]]:
    """Analyzes a source dataset to define a class-balanced generation strategy.

    This function inspects the source data, determines if it is structured as
    class directories or a COCO-style JSON manifest, and calculates how many
    new images to generate for each class to meet the target percentage.

    Args:
        source_train_path: Path to the source training data (either a directory
          of class folders or a single .json file).
        generation_percentage: The percentage of the total dataset size to
          generate as new images.

    Returns:
        A tuple containing:
        - A dictionary mapping class names to the number of images to generate.
        - A string indicating the detected source format ('directory' or 'json').
        Returns (None, None) on failure.
    """
    target_counts_per_class: Dict[str, int] = {}
    images_by_class = defaultdict(int)
    total_images = 0
    source_format = None

    if not source_train_path.exists():
        logger.error(f"Source training path does not exist: {source_train_path}")
        return None, None

    # --- Logic for Directory-based (ImageFolder-style) dataset ---
    if source_train_path.is_dir():
        source_format = 'directory'
        class_names = sorted([d.name for d in source_train_path.iterdir() if d.is_dir()])
        if not class_names:
            logger.error(f"No class subdirectories found in {source_train_path}.")
            return None, source_format
        
        images_by_class = {name: len(list((source_train_path / name).glob('*'))) for name in class_names}
        total_images = sum(images_by_class.values())
        logger.info(f"Directory dataset detected. Found {total_images} images across {len(class_names)} classes.")

        # DEBUG: Stampa l'ordine delle classi e gli ID assegnati
        logger.debug("Mappatura Classi per dataset basato su directory:")
        for i, class_name in enumerate(class_names):
            logger.debug(f"  Classe: '{class_name}' -> Indice: {i}")

    # --- Logic for JSON-based (COCO-style) dataset ---
    elif source_train_path.name.endswith('.json'):
        source_format = 'json'
        try:
            with open(source_train_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            
            cat_id_to_name = {cat['id']: cat['name'] for cat in manifest['categories']}

            sorted_categories = sorted(manifest['categories'], key=lambda x: x['name'])
            class_names_from_json = [cat['name'] for cat in sorted_categories]

            for ann in manifest['annotations']:
                if (class_name := cat_id_to_name.get(ann['category_id'])):
                    images_by_class[class_name] += 1
            
            total_images = len(manifest['images'])
            logger.info(f"JSON dataset detected. Found {total_images} images across {len(images_by_class)} classes.")
            
            # DEBUG: Stampa l'ordine delle classi e gli ID assegnati
            logger.debug("Mappatura Classi per dataset basato su JSON (ordinato per nome):")
            for i, class_name in enumerate(class_names_from_json):
                logger.debug(f"  Classe: '{class_name}' -> Indice: {i}")

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to process JSON file {source_train_path}: {e}", exc_info=True)
            return None, source_format
    else:
        logger.error(f"Unrecognized source data type: {source_train_path}. Not a directory or .json file.")
        return None, None

        

    # --- Create balanced generation plan (common to both formats) ---
    num_to_generate_total = int(total_images * (generation_percentage / 100.0))
    if num_to_generate_total > 0 and images_by_class:
        num_classes = len(images_by_class)
        base_num = num_to_generate_total // num_classes
        remainder = num_to_generate_total % num_classes
        
        class_keys = list(images_by_class.keys())
        target_counts_per_class = {cls: base_num for cls in class_keys}
        for i in range(remainder):
            target_counts_per_class[class_keys[i]] += 1
            
    logger.info(f"Generation plan (new images per class): {target_counts_per_class}")
    return target_counts_per_class, source_format


# =============================================================================
# 4. WORKFLOW STAGE 2: IMAGE SYNTHESIS
# =============================================================================

def resolve_synthesis_engine(engine_type: str) -> Optional[Callable]:
    """Resolves the requested synthesis engine function from a string key.

    This factory function maps a string from the configuration file to the
    actual Python function responsible for image generation.

    Args:
        engine_type: The key for the engine (e.g., 'lora_generate').

    Returns:
        The callable synthesis function, or None if the key is invalid.
    """
    engine_map = {
        'lora_generate': synthesize_with_finetuned_lora,
        'classic_manual': synthesize_from_manual_policy,
        'stylegan_generate': synthesize_using_gan,
    }
    engine = engine_map.get(engine_type)
    if not engine:
        logger.error(f"Invalid synthesis engine type '{engine_type}' specified in config.")
    return engine


def synthesize_new_images(
    job_cfg: DictConfig,
    target_counts: Dict[str, int],
    source_train_path: Path
) -> List[Tuple[str, Image.Image, str]]:
    """Generates images using a fine-tuned LoRA/Diffusion model.

    This engine loads a Stable Diffusion pipeline, applies LoRA weights from a
    checkpoint, and generates images from class-specific text prompts. It uses
    a global cache to avoid reloading the same model across different classes.

    Args:
        params: Configuration parameters for the engine.
        num_to_generate: The number of images to generate.
        class_label: The class name to generate images for.
        **kwargs: Catches unused arguments.

    Returns:
        A list of (filename, PIL.Image, class_label) tuples.
    """
    synthesis_engine = resolve_synthesis_engine(job_cfg.type)
    if not synthesis_engine:
        return []

    all_synthesized_samples: List[Tuple[str, Image.Image, str]] = []
    source_format = 'directory' if source_train_path.is_dir() else 'json'
    class_to_images_map = {}

    # If using classic augmentation on a JSON dataset, pre-load the image paths for each class.
    if job_cfg.type == 'classic_manual' and source_format == 'json':
        logger.info("Pre-loading image paths from JSON manifest for classic augmentation...")
        image_dir_path_str = job_cfg.params.get('coco_image_source_dir')
        if not image_dir_path_str:
            logger.error("Classic augmentation on JSON requires 'coco_image_source_dir' in job params.")
            return []
        
        # Resolve path from original CWD to handle Hydra's directory changes.
        image_dir = Path(get_original_cwd()) / image_dir_path_str
        with open(source_train_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        cat_id_to_name = {cat['id']: cat['name'] for cat in manifest.get('categories', [])}
        images_by_id = {img['id']: img['file_name'] for img in manifest.get('images', [])}
        
        class_names_in_manifest = sorted([cat['name'] for cat in manifest.get('categories', [])])
        for class_name in class_names_in_manifest: # Usa l'ordine ordinato
            class_to_images_map[class_name] = []

        for ann in manifest['annotations']:
            class_name = cat_id_to_name.get(ann['category_id'])
            file_name = images_by_id.get(ann['image_id'])
            if class_name and file_name:
                class_to_images_map[class_name].append(image_dir / file_name)

    # Create a mapping from class name to a numerical index for conditional GANs.

    sorted_class_names = sorted(target_counts.keys())
    class_to_idx = {name: i for i, name in enumerate(sorted_class_names)}

    # DEBUG: Stampa la mappatura finale usata per la generazione
    logger.info("Mappatura finale Class-to-Index per la generazione:")
    for class_name, idx in class_to_idx.items():
        logger.info(f"  Classe: '{class_name}' -> Indice: {idx}")

    # Iterate through the generation plan and synthesize images.
    for class_label, num_to_generate in target_counts.items():
        if num_to_generate > 0:
            logger.info(f"Synthesizing {num_to_generate} images for class '{class_label}'...")
            
            engine_args = {
                "params": job_cfg.params,
                "num_to_generate": num_to_generate,
                "class_label": class_label,
            }

            # Add engine-specific arguments.
            if job_cfg.type == 'stylegan_generate':
                engine_args["class_idx"] = class_to_idx.get(class_label)
            elif job_cfg.type == 'classic_manual':
                if source_format == 'directory':
                    engine_args["source_images"] = list((source_train_path / class_label).glob('*'))
                else:  # 'json'
                    engine_args["source_images"] = class_to_images_map.get(class_label, [])
            
            generated_tuples = synthesis_engine(**engine_args)
            all_synthesized_samples.extend(generated_tuples)
            
    return all_synthesized_samples


# --- Synthesis Engine Implementations ---

def _generate_training_consistent_prompt(class_name: str) -> str:
    """
    Generates a standardized text prompt for a given class name.

    Args:
        class_name: The name of the class (e.g., 'cancer').

    Returns:
        A formatted string to be used as a prompt for diffusion models.
    """
    class_name_lower = class_name.lower().strip().replace('_', ' ')
    if class_name_lower == 'cancer':
        prompt = "high quality picture of oral cancer"
    elif class_name_lower in ['no cancer', 'non cancer']:
        prompt = "high quality picture of a healthy oral cavity"
    else:
        prompt = f"high quality picture of oral {class_name_lower} lesion"
    return prompt


def synthesize_with_finetuned_lora(
    params: DictConfig,
    num_to_generate: int,
    class_label: str,
    **kwargs
) -> List[Tuple[str, Image.Image, str]]:
    """
    Generates images using a fine-tuned LoRA/Diffusion model.

    This engine loads a Stable Diffusion pipeline, applies LoRA weights from a
    checkpoint, and generates images based on class-specific prompts. It caches
    the loaded pipeline to improve performance when processing multiple classes.

    Args:
        params: Configuration parameters for the engine from the config file.
        num_to_generate: The number of images to generate for the class.
        class_label: The name of the class to generate images for.
        **kwargs: Catches any unused arguments.

    Returns:
        A list of (filename, PIL.Image, class_label) tuples.
    """
    logger.info(f"--- Initializing LoRA Generation Engine for Class: {class_label} ---")
    if not _LORA_MODEL_AVAILABLE:
        logger.error("LoraDiffusionModel not available. Cannot proceed.")
        return []

    # Resolve checkpoint path to be absolute.
    ckpt_path = Path(params.ckpt_path).resolve()
    
    # Use a global cache to avoid reloading the same model.
    cache_key = str(ckpt_path)
    pipeline = _LORA_PIPELINE_CACHE.get(cache_key)

    if pipeline is None:
        logger.info(f"Pipeline not in cache. Loading model from disk: {ckpt_path}")
        device = params.compute.get("device", "cuda")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            logger.warning("CUDA not available, falling back to CPU.")
        
        try:
            pl_model = LoraDiffusionModel.load_from_checkpoint(str(ckpt_path), map_location=device, strict=False)
            pl_model.eval()
            
            scheduler_type = params.generation.get("scheduler_type", "dpm").lower()
            scheduler_class = DDIMScheduler if scheduler_type == "ddim" else DPMSolverMultistepScheduler
            scheduler = scheduler_class.from_pretrained(params.base_model_name, subfolder="scheduler")
            
            pipeline = StableDiffusionPipeline(
                vae=pl_model.vae, text_encoder=pl_model.text_encoder,
                tokenizer=pl_model.tokenizer, unet=pl_model.unet,
                scheduler=scheduler, safety_checker=None, feature_extractor=None,
            ).to(device)

            _LORA_PIPELINE_CACHE[cache_key] = pipeline
            logger.info(f"Pipeline for '{ckpt_path.name}' has been cached.")
        except Exception as e:
            logger.exception(f"Failed to load LoRA checkpoint or create pipeline: {e}")
            return []
    else:
        logger.info(f"Loaded LoRA pipeline from cache for '{ckpt_path.name}'.")

    # Generate images for the current class.
    synthesized_samples: List[Tuple[str, Image.Image, str]] = []
    prompt = _generate_training_consistent_prompt(class_label)
    logger.info(f"Using prompt for '{class_label}': '{prompt}'")
    
    master_seed = params.get("seed", -1)
    if master_seed == -1:
        master_seed = random.randint(0, 2**32 - 1)
    
    inference_dtype = getattr(torch, params.compute.get("dtype", "float16"))
    for i in tqdm(range(num_to_generate), desc=f"Generating '{class_label}'", unit="image"):
        current_seed = master_seed + i
        generator = torch.Generator(device=pipeline.device).manual_seed(current_seed)
        try:
            with torch.inference_mode(), torch.autocast(device_type=pipeline.device.type, dtype=inference_dtype):
                result_image = pipeline(
                    prompt=prompt,
                    negative_prompt=params.get("negative_prompt", ""),
                    num_inference_steps=params.generation.steps,
                    guidance_scale=params.generation.guidance_scale,
                    generator=generator
                ).images[0]
            
            filename = f"lora_{class_label.replace(' ', '_').lower()}_{current_seed}.png"
            synthesized_samples.append((filename, result_image, class_label))
        except Exception as e:
            logger.exception(f"Failed during generation for seed {current_seed}: {e}")
            continue

    return synthesized_samples


def synthesize_from_manual_policy( # not used in the current config
    params: DictConfig,
    num_to_generate: int,
    class_label: str,
    source_images: List[Path],
    **kwargs
) -> List[Tuple[str, Image.Image, str]]:
    """Synthesizes images using a user-defined policy of classic transforms.

    This engine applies a composition of torchvision transforms to randomly
    selected source images from the specified class.

    Args:
        params: Configuration parameters, containing a 'transforms' block.
        num_to_generate: The number of augmented images to create.
        class_label: The name of the current class.
        source_images: A list of Paths to the source images for this class.
        **kwargs: Catches unused arguments.

    Returns:
        A list of (filename, PIL.Image, class_label) tuples.
    """
    if not source_images:
        logger.error(f"No source images found for classic augmentation for class '{class_label}'.")
        return []
        
    augment_transforms = hydra.utils.instantiate(params.transforms)
    logger.info(f"Applying manual classic augmentation policy to class '{class_label}'.")
    
    synthesized_samples: List[Tuple[str, Image.Image, str]] = []
    for i in tqdm(range(num_to_generate), desc=f"Classic Aug: {class_label}", unit="image"):
        try:
            source_img_path = random.choice(source_images)
            with Image.open(source_img_path) as img:
                img_rgb = img.convert("RGB")
                
            augmented_img = augment_transforms(img_rgb)
            filename = f"classic_{class_label.replace(' ', '_').lower()}_{i+1}.png"
            synthesized_samples.append((filename, augmented_img, class_label))
            logger.debug(f"Created augmented sample: {filename} from source {source_img_path.name}")
        except Exception as e:
            logger.error(f"Error during classic augmentation for source {source_img_path.name}: {e}", exc_info=True)
            
    return synthesized_samples


def synthesize_using_gan(
    params: DictConfig,
    num_to_generate: int,
    class_label: str,
    class_idx: int,
    **kwargs
) -> List[Tuple[str, Image.Image, str]]:
    """Synthesizes images conditionally using an external StyleGAN script.

    This function is a wrapper around a command-line tool (e.g., StyleGAN3's
    `gen_images.py`). It builds the command from config parameters and executes
    it as a subprocess to generate images.

    Args:
        params: Configuration parameters for the StyleGAN engine.
        num_to_generate: The number of images to generate.
        class_label: The name of the current class.
        class_idx: The numerical index of the class for conditional generation.
        **kwargs: Catches unused arguments.

    Returns:
        A list of (filename, PIL.Image, class_label) tuples.
    """
    logger.info(f"--- Initializing Conditional StyleGAN Engine for Class: '{class_label}' (Index: {class_idx}) ---")
    if not all(k in params for k in ['stylegan_repo_path', 'network_pkl']):
        logger.error("StyleGAN config is missing 'stylegan_repo_path' or 'network_pkl'.")
        return []

    repo_path = Path(params.stylegan_repo_path)
    gen_script_name = 'gen_images.py'
    gen_script_path = repo_path / gen_script_name
    network_pkl = Path(params.network_pkl).resolve() # Resolve to absolute path

    if not gen_script_path.is_file():
        logger.error(f"StyleGAN3 '{gen_script_name}' not found at: {gen_script_path}")
        return []
    if not network_pkl.is_file():
        logger.error(f"StyleGAN checkpoint not found at: {network_pkl}")
        return []
        
    if class_idx is None:
        logger.error(f"Conditional generation requires a class_idx, but none was provided for class '{class_label}'.")
        return []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        increment_offset = 3
        user_seed = params.get('seed', -1)
        if user_seed is None or user_seed == -1:
            base_seed = random.randint(0, 1_000_000)
        else:
            class_offset = hash(class_label) % 10000
            base_seed = int(user_seed) + class_offset

        generated_seeds = []
        for i in range(num_to_generate):
            generated_seeds.append(base_seed + (i * increment_offset))

        seed_range = ','.join(map(str, generated_seeds))

        # Build the command-line arguments for gen_images.py
        command = [
            sys.executable, gen_script_name,
            f'--network={network_pkl}',
            f'--outdir={temp_path}',
            f'--seeds={seed_range}',
            f'--class={class_idx}'
        ]
        optional_cli_params = {
            'truncation_psi': '--trunc',
            'noise_mode': '--noise-mode',
            'translate': '--translate',
            'rotate': '--rotate',
        }
        for param_key, cli_flag in optional_cli_params.items():
            param_value = params.get(param_key)
            if param_value is not None:
                command.append(f'{cli_flag}={param_value}')
        
        logger.info(f"Executing command: {' '.join(str(c) for c in command)}")
        try:
            # Execute the script with the CWD set to the repo path.
            # This ensures any relative paths within the StyleGAN3 code work correctly.
            subprocess.run(
                command, check=True, capture_output=True, text=True, cwd=repo_path
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"The '{gen_script_name}' script failed. Stderr:\n{e.stderr}")
            return []

        # Load the generated images from the temporary directory.
        generated_files = sorted(list(temp_path.glob('*.png')))
        if len(generated_files) != num_to_generate:
            logger.warning(f"Expected {num_to_generate} images, but found {len(generated_files)}.")

        synthesized_samples: List[Tuple[str, Image.Image, str]] = []
        for i, img_path in enumerate(tqdm(generated_files, desc=f"Loading GAN images for '{class_label}'")):
            with Image.open(img_path) as pil_img:
                pil_img.load()
            filename = f"stylegan_{class_label.replace(' ', '_').lower()}_{base_seed + i}.png"
            synthesized_samples.append((filename, pil_img, class_label))
    
    return synthesized_samples


# =============================================================================
# 5. WORKFLOW STAGE 3: DATASET MATERIALIZATION
# =============================================================================

def _add_new_entries_to_manifest(
    manifest: Dict,
    synthesized_samples: List[Tuple[str, Image.Image, str]],
    image_dir_name: str
):
    """Adds metadata for newly generated images to a COCO-style manifest.

    Args:
        manifest: The manifest dictionary (loaded from a .json file).
        synthesized_samples: The list of newly generated image tuples.
    """
    if not all(k in manifest for k in ["images", "annotations", "categories"]):
        logger.error("Manifest is not in COCO-like format. Skipping update.")
        return

    cat_name_to_id = {cat['name']: cat['id'] for cat in manifest['categories']}
    max_img_id = max([img['id'] for img in manifest['images']], default=0)
    max_ann_id = max([ann['id'] for ann in manifest['annotations']], default=0)
    logger.info(f"Adding {len(synthesized_samples)} new entries to JSON manifest.")

    for i, (filename, pil_img, class_label) in enumerate(synthesized_samples):
        if class_label not in cat_name_to_id:
            logger.warning(f"Class '{class_label}' not in manifest categories. Skipping {filename}.")
            continue

        new_img_id = max_img_id + i + 1
        new_ann_id = max_ann_id + i + 1
        
        image_entry = {
            "id": new_img_id, "width": pil_img.width, "height": pil_img.height,
            "file_name": f"{filename}"
        }
        manifest['images'].append(image_entry)
        
        annotation_entry = {
            "id": new_ann_id, "image_id": new_img_id,
            "category_id": cat_name_to_id[class_label], "bbox": [0, 0, pil_img.width, pil_img.height],
            "area": pil_img.width * pil_img.height, "iscrowd": 0,
        }
        manifest['annotations'].append(annotation_entry)


def materialize_augmented_dataset(
    job_cfg: DictConfig,
    source_cfg: DictConfig,
    source_format: str,
    synthesized_samples: List[Tuple]
):
    """Creates the final augmented dataset on disk.

    This function first copies the original dataset to a new output directory.
    It then saves the newly synthesized images into the appropriate location,
    updating the COCO JSON manifest if one exists.

    Args:
        job_cfg: Configuration for the current augmentation job.
        source_cfg: Configuration for the source dataset.
        source_format: The detected format ('directory' or 'json').
        synthesized_samples: A list of (filename, PIL.Image, class_label) tuples.
    """
    # Resolve paths from original CWD to handle Hydra's directory changes.
    original_cwd = Path(get_original_cwd())
    source_root = original_cwd / source_cfg.base_path / source_cfg.name / source_cfg.input_augmentation_type
    output_root = original_cwd / source_cfg.base_path / source_cfg.name / job_cfg.output_augmentation_type
    
    logger.info(f"Materializing augmented dataset at: {output_root}")
    
    logger.info(f"Copying original dataset from {source_root} to {output_root}...")
    if output_root.exists():
        logger.warning(f"Output directory {output_root} already exists. Overwriting.")
        shutil.rmtree(output_root)
    shutil.copytree(source_root, output_root)
    
    if not synthesized_samples:
        logger.info("No new images were synthesized. Copied dataset is final.")
        return

    # Save newly synthesized samples based on the original format.
    if source_format == 'directory':
        logger.info(f"Adding {len(synthesized_samples)} new images to directory-based training set...")
        train_output_dir = output_root / "train"
        for filename, pil_img, class_label in tqdm(synthesized_samples, desc="Saving to directories", unit="image"):
            class_output_dir = train_output_dir / class_label
            class_output_dir.mkdir(exist_ok=True)

            if pil_img.mode == 'RGBA':
                pil_img = pil_img.convert('RGB')

            jpg_filename = Path(filename).with_suffix('.jpg').name
            pil_img.save(class_output_dir / jpg_filename, format='JPEG', quality=95)
    
    elif source_format == 'json':
        logger.info(f"Adding {len(synthesized_samples)} new images to JSON-based training set...")
        output_img_dir = output_root / source_cfg.subfolder
        output_img_dir.mkdir(exist_ok=True)

        for i, (filename, pil_img, metadata) in enumerate(tqdm(synthesized_samples, desc="Saving image files", unit="image")):
            if pil_img.mode == 'RGBA':
                pil_img = pil_img.convert('RGB')
            
            jpg_filename = Path(filename).with_suffix('.jpg').name
            pil_img.save(output_img_dir / jpg_filename, format='JPEG', quality=95)

            # Update the entry in the original list
            synthesized_samples[i] = (jpg_filename, pil_img, metadata)

        train_json_path = output_root / "train.json"
        if train_json_path.exists():
            with open(train_json_path, 'r+') as f:
                manifest = json.load(f)
                _add_new_entries_to_manifest(manifest, synthesized_samples, source_cfg.name)
                f.seek(0)
                f.truncate()
                json.dump(manifest, f, indent=4)
        else:
            logger.warning(f"Could not find {train_json_path} to update with new image metadata.")

    logger.info(f"Dataset augmentation complete. Final dataset at: {output_root}")



# =============================================================================
# 6. MAIN ORCHESTRATION
# =============================================================================

@hydra.main(config_path="config", config_name="offline_data_augmentation", version_base=None)
def main(cfg: DictConfig):
    """
    Main execution entry point for the augmentation script.
    
    This function orchestrates the entire 'DEFINE -> SYNTHESIZE -> MATERIALIZE'
    workflow for each job specified in the configuration file.
    
    Args:
        cfg: The configuration object loaded by Hydra.
    """
    logger.info("Starting offline data augmentation process...")
    if not cfg.get("augmentation_jobs"):
        logger.critical("Config error: 'augmentation_jobs' list not found."); return

    source_cfg = cfg.source_dataset
    original_cwd = Path(get_original_cwd())
    
    for i, job_cfg in enumerate(cfg.augmentation_jobs):
        job_name = job_cfg.get('name', f'Job {i+1}')
        logger.info(f"--- [Job {i+1}/{len(cfg.augmentation_jobs)}] Starting: {job_name} ---")
        if not job_cfg.get('enabled', True):
            logger.warning(f"Job '{job_name}' is disabled in config. Skipping."); continue
        
        try:
            # --- STAGE 1: DEFINE STRATEGY ---
            base_data_path = original_cwd / source_cfg.base_path / source_cfg.name / source_cfg.input_augmentation_type
            
            # Smartly determine the path to the training data (directory or .json file).
            train_dir_path = base_data_path / 'train'
            source_path_for_strategy = train_dir_path if train_dir_path.is_dir() else base_data_path / "train.json"

            target_counts, source_format = define_augmentation_strategy(
                source_train_path=source_path_for_strategy,
                generation_percentage=job_cfg.generation_percentage
            )
            
            if not target_counts:
                logger.warning("Generation plan is empty. Nothing to generate for this job.")
                materialize_augmented_dataset(job_cfg, source_cfg, 'directory', [])
                continue

            # --- STAGE 2: SYNTHESIZE ---
            synthesized_samples = synthesize_new_images(
                job_cfg=job_cfg,
                target_counts=target_counts,
                source_train_path=source_path_for_strategy
            )

            # --- STAGE 3: MATERIALIZE ---
            materialize_augmented_dataset(
                job_cfg=job_cfg,
                source_cfg=source_cfg,
                source_format=source_format,
                synthesized_samples=synthesized_samples
            )

            logger.info(f"--- [Job {i+1}/{len(cfg.augmentation_jobs)}] Finished: {job_name} ---")
        except Exception as e:
            logger.exception(f"--- [Job {i+1}/{len(cfg.augmentation_jobs)}] FAILED UNEXPECTEDLY: {e} ---")
            
    logger.info("All augmentation jobs completed.")


if __name__ == "__main__":
    main()