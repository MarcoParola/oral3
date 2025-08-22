# =============================================================================
# IMPORTS
# =============================================================================

# --- Standard Library Imports ---
import logging
import random
import shutil
import subprocess
import sys
import tempfile 
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Tuple

# --- Third-party Imports ---
import hydra
import torch
import numpy as np
from diffusers import (DPMSolverMultistepScheduler,
                       StableDiffusionImg2ImgPipeline, StableDiffusionPipeline)
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# --- Local Application Imports ---
try:
    from src.data.stable_diffusion.dataset import DiffusionDataset
except ImportError:
    print("Error: Could not import DiffusionDataset. Ensure 'src/data/stable_diffusion/dataset.py' exists.")
    DiffusionDataset = None

try:
    from src.models.stable_diffusion import LoraDiffusionModel
    _LORA_MODEL_AVAILABLE = True
except ImportError:
    print("WARNING: LoraDiffusionModel not found. LoRA-based generation will be unavailable.")
    LoraDiffusionModel = None
    _LORA_MODEL_AVAILABLE = False

# --- StyleGAN3 Library Imports ---
_STYLEGAN_AVAILABLE = False
dnnlib = None
legacy = None

# =============================================================================
# SCRIPT SETUP
# =============================================================================
def setup_logger(name: str, log_file: Path = None, level=logging.INFO) -> logging.Logger:
    """Configures and returns a logger instance for console or file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent propagation to the root logger to avoid duplicate messages.
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler(log_file) if log_file else logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

# =============================================================================
# AUGMENTATION WORKFLOW 
# =============================================================================
class DatasetAugmenter:
    """
    Manages the data augmentation workflow for a single configured job.

    This class encapsulates all logic for a single augmentation task, from
    model loading to image generation and file system interaction. Each instance
    is responsible for one entry in the `augmentation_jobs` configuration list.
    """
    def __init__(self, job_cfg: DictConfig, dataset: DiffusionDataset, logger: logging.Logger, output_dir: Path):
        self.job_cfg = job_cfg
        self.dataset = dataset
        self.logger = logger
        self.output_dir = output_dir
        self.class_names = self.dataset.class_names

        if not self.class_names:
            raise ValueError("No classes were found in the source dataset.")
        
        # Define how many images to generate for each class.
        num_to_gen_per_class = self.job_cfg.get('num_per_class', 5)
        self.generation_plan = {cls: num_to_gen_per_class for cls in self.class_names}
        self.logger.info(f"Generation plan created: {self.generation_plan}")
        self.pipeline = None
        self.stylegan_model = None

    def run(self):
        """Executes the full generation and saving pipeline for this job."""
        self.logger.info("Starting image generation and saving with buffer...")
        train_output_dir = self.output_dir / "train"
        if train_output_dir.exists(): shutil.rmtree(train_output_dir)
        train_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure an optional directory for saving side-by-side comparisons
        # in image-to-image tasks.
        comparison_output_dir = None
        is_img2img_job = self.job_cfg.type == 'lora_generate_img2img'
        if is_img2img_job and self.job_cfg.params.get('save_comparison', False):
            comparison_output_dir = self.output_dir / "comparisons"
            if comparison_output_dir.exists(): shutil.rmtree(comparison_output_dir)
            comparison_output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Comparison saving enabled. Outputting to: {comparison_output_dir}")
        
        # Use a buffer to reduce frequent disk I/O operations.
        buffer = []
        buffer_size = 10
        image_counter = 0
        for generated_item in self._generate_images():
            if generated_item:
                buffer.append(generated_item)
                image_counter += 1
                if len(buffer) >= buffer_size:
                    self.logger.info(f"Saving buffer of {len(buffer)} images...")
                    self._save_image_buffer(buffer, train_output_dir, comparison_output_dir)
                    buffer.clear()

        # Save any remaining items in the buffer after generation is complete.
        if buffer:
            self.logger.info(f"Saving final buffer of {len(buffer)} images...")
            self._save_image_buffer(buffer, train_output_dir, comparison_output_dir)
            buffer.clear()
        self.logger.info(f"Job finished. Total images saved: {image_counter}")

    def _generate_images(self) -> Generator[Tuple, None, None]:
        """Dispatches to the correct generation engine, yielding generated items."""
        engine_map = {
            'lora_generate_txt2img': self._generate_with_lora_txt2img,
            'lora_generate_img2img': self._generate_with_lora_img2img,
            'manual_augment': self._apply_manual_augmentations,
            'stylegan3_generate': self._generate_with_stylegan3
        }
        synthesis_engine = engine_map.get(self.job_cfg.type)
        if not synthesis_engine:
            self.logger.error(f"Invalid synthesis engine type '{self.job_cfg.type}' specified in config.")
            return

        # Progressive seed to ensure unique generation per class/image
        job_start_seed = self.job_cfg.params.seed
        class_seed_offset = 0
        for class_label, num_to_generate in self.generation_plan.items():
            if num_to_generate > 0:
                self.logger.info(f"Running engine '{self.job_cfg.type}' for {num_to_generate} images of class '{class_label}'...")
                current_params = self.job_cfg.params.copy()
                current_params.seed = job_start_seed + class_seed_offset
                yield from synthesis_engine(params=current_params, num_to_generate=num_to_generate, class_label=class_label)
                class_seed_offset += num_to_generate
    
    def _create_comparison_image(self, original_img: Image.Image, generated_img: Image.Image) -> Image.Image:
        """Creates a new image by placing the original and generated images side-by-side."""
        if original_img.size != generated_img.size:
            generated_img = generated_img.resize(original_img.size, Image.Resampling.LANCZOS)
        dst = Image.new('RGB', (original_img.width * 2, original_img.height))
        dst.paste(original_img, (0, 0))
        dst.paste(generated_img, (original_img.width, 0))
        return dst

    def _save_image_buffer(self, buffer: List[Tuple], train_output_dir: Path, comparison_output_dir: Path = None):
        """Saves a buffer of generated images and optional comparisons."""
        is_img2img_job = self.job_cfg.type == 'lora_generate_img2img'
        save_comparisons_enabled = self.job_cfg.params.get('save_comparison', False)
        for data_item in buffer:
            if is_img2img_job:
                filename, generated_img, original_img, class_label = data_item
                pil_img_to_save = generated_img

                # If comparison saving is enabled, create and save the side-by-side image.
                if save_comparisons_enabled and comparison_output_dir:
                    comparison_img = self._create_comparison_image(original_img, generated_img)
                    class_comparison_dir = comparison_output_dir / class_label
                    class_comparison_dir.mkdir(exist_ok=True)
                    comparison_filename = f"compare_{Path(filename).stem}.jpg"
                    comparison_img.save(class_comparison_dir / comparison_filename, format='JPEG', quality=90)
            else:
                filename, pil_img_to_save, class_label = data_item
            
            # Save the primary generated image to the training output directory.
            class_train_dir = train_output_dir / class_label
            class_train_dir.mkdir(exist_ok=True)
            if pil_img_to_save.mode != 'RGB':
                pil_img_to_save = pil_img_to_save.convert('RGB')
            pil_img_to_save.save(class_train_dir / Path(filename).with_suffix('.jpg').name, format='JPEG', quality=95)

    def _get_lora_pipeline(self, params: DictConfig, pipeline_class):
        """Loads and prepares the LoRA diffusion pipeline for the current job."""
        if self.pipeline: return self.pipeline
        try:
            device = params.compute.get("device", "cuda")
            pl_model = LoraDiffusionModel.load_from_checkpoint(params.ckpt_path, map_location=device, strict=False)
            scheduler = DPMSolverMultistepScheduler.from_pretrained(params.base_model_name, subfolder="scheduler")
            self.pipeline = pipeline_class(
                vae=pl_model.vae, 
                text_encoder=pl_model.text_encoder, 
                tokenizer=pl_model.tokenizer, 
                unet=pl_model.unet, 
                scheduler=scheduler, 
                safety_checker=None, 
                feature_extractor=None
            ).to(device)
            self.logger.info(f"Successfully loaded LoRA pipeline for checkpoint: {Path(params.ckpt_path).name}")
            return self.pipeline
        except Exception as e:
            self.logger.exception(f"Failed to load LoRA checkpoint: {e}"); return None
            
    def _instantiate_transforms_from_cfg(self, cfg: ListConfig) -> transforms.Compose:
        """Instantiates a torchvision transforms pipeline from a Hydra config."""
        transform_list = []
        if not cfg: return transforms.Compose(transform_list)
        interpolation_map = {
            "NEAREST": transforms.InterpolationMode.NEAREST,
            "BILINEAR": transforms.InterpolationMode.BILINEAR,
            "BICUBIC": transforms.InterpolationMode.BICUBIC,
            "LANCZOS": transforms.InterpolationMode.LANCZOS
        }
        for t_cfg in cfg:
            # This logic handles the `interpolation` argument for Resize transforms.
            if '_target_' in t_cfg and 'Resize' in t_cfg._target_ and 'interpolation' in t_cfg and isinstance(t_cfg.interpolation, str):
                t_cfg.interpolation = interpolation_map.get(t_cfg.interpolation.upper(), transforms.InterpolationMode.BICUBIC)
            try:
                transform_list.append(hydra.utils.instantiate(t_cfg))
            except Exception as e:
                self.logger.error(f"Failed to instantiate transform from config: {t_cfg}. Error: {e}")
        return transforms.Compose(transform_list)

    def _apply_manual_augmentations(self, params: DictConfig, num_to_generate: int, class_label: str) -> Generator[Tuple, None, None]:
        """Applies a chain of traditional vision transforms to source images."""
        source_items = self.dataset.class_to_items_map.get(class_label, [])
        if not source_items:
            self.logger.error(f"No source items for class '{class_label}' to apply manual augmentations."); return
        try:
            transform_pipeline = self._instantiate_transforms_from_cfg(params.transforms)
            self.logger.info(f"Applying manual transform pipeline for class '{class_label}': {transform_pipeline}")
        except Exception as e:
            self.logger.error(f"Failed to create transform pipeline: {e}"); return
        for i in tqdm(range(num_to_generate), desc=f"Augmenting '{class_label}'"):
            source_path, _, _ = random.choice(source_items)
            try:
                original_image = Image.open(source_path).convert("RGB")
                augmented_image = transform_pipeline(original_image)
                filename = f"manual_{class_label.replace(' ', '_').lower()}_{Path(source_path).stem}_{i}.png"
                yield (filename, augmented_image, class_label)
            except Exception as e:
                self.logger.error(f"Failed to apply manual augmentation to {source_path}: {e}")
                
   
    def _generate_with_stylegan3(self, params: DictConfig, num_to_generate: int, class_label: str) -> Generator[Tuple, None, None]:
        """Generates images by calling the external StyleGAN3 script in a subprocess."""
        if 'stylegan_repo_path' not in params:
             self.logger.error("Global 'stylegan3_repo_path' is not defined in the YAML. Skipping GAN job."); return
        
        repo_path = Path(get_original_cwd()) / params.stylegan_repo_path
        gen_script = repo_path / 'gen_images.py'
        if not gen_script.is_file():
            self.logger.error(f"StyleGAN script not found at: {gen_script}"); return

        network_pkl = Path(params.ckpt_path).resolve()
        class_idx = self.dataset.class_to_idx.get(class_label)
        if class_idx is None:
            self.logger.error(f"Class '{class_label}' not found in dataset's class map for StyleGAN generation."); return
        
        # Use a temporary directory to store images generated by the subprocess.
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            master_seed = params.seed
            seeds = ','.join(map(str, range(master_seed, master_seed + num_to_generate)))

            # The script is executed as a module (`-m`) to ensure correct relative imports
            # within the StyleGAN3 project structure.
            module_path = params.stylegan_repo_path.replace('/', '.').replace('\\', '.') + '.gen_images'

            command = [sys.executable, '-m', module_path, f'--network={network_pkl}', f'--outdir={temp_path}', f'--seeds={seeds}', f'--class={class_idx}']
            if (psi := params.get('truncation_psi')) is not None:
                command.append(f'--trunc={psi}')
            
            try:
                self.logger.info(f"Executing StyleGAN generation as module: '{module_path}'")

                # The PYTHONPATH is modified for the subprocess to correctly locate
                # StyleGAN's internal dependencies (e.g., dnnlib)

                env = os.environ.copy()
                repo_path_abs = str(repo_path.resolve())
                env['PYTHONPATH'] = f"{repo_path_abs}{os.pathsep}{env.get('PYTHONPATH', '')}"

                process = subprocess.Popen(command, cwd=get_original_cwd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', env=env)
                
                # Stream the subprocess output to the main console in real-time.
                for line in iter(process.stdout.readline, ''):
                    print(line, end='')
                
                process.stdout.close()
                return_code = process.wait()
                if return_code:
                    raise subprocess.CalledProcessError(return_code, command)

            except subprocess.CalledProcessError:
                self.logger.error(f"StyleGAN script failed. See console output above for details."); return
            
            # Process the generated images from the temporary directory.
            for img_path in sorted(list(temp_path.glob('*.png'))):
                try:
                    with Image.open(img_path) as pil_img:
                        pil_img.load()
                    original_seed = Path(img_path).stem.replace('seed', '')
                    filename = f"stylegan3_{class_label.replace(' ', '_').lower()}_{original_seed}.png"
                    yield (filename, pil_img.copy(), class_label)
                except Exception as e:
                    self.logger.error(f"Failed to process generated StyleGAN image {img_path}: {e}")

    def _generate_with_lora_txt2img(self, params: DictConfig, num_to_generate: int, class_label: str) -> Generator[Tuple, None, None]:
        """Generates images from text prompts using a LoRA model."""
        if not _LORA_MODEL_AVAILABLE: return
        pipeline = self._get_lora_pipeline(params, StableDiffusionPipeline)
        if pipeline is None: return

        prompts = [self.dataset._generate_prompt_for_class(class_label) for _ in range(num_to_generate)]
        image_size = params.generation.get('image_size', 256)

        for i, prompt in enumerate(tqdm(prompts, desc=f"Generating '{class_label}'")):
            current_seed = params.seed + i
            generator = torch.Generator(device=pipeline.device).manual_seed(current_seed)
            try:
                with torch.inference_mode(), torch.autocast(device_type=pipeline.device.type, dtype=getattr(torch, params.compute.dtype)):
                    result_image = pipeline(prompt=prompt, negative_prompt=params.get("negative_prompt", ""), num_inference_steps=params.generation.steps, guidance_scale=params.generation.guidance_scale, generator=generator, height=image_size, width=image_size).images[0]
                filename = f"txt2img_{class_label.replace(' ', '_').lower()}_{current_seed}.png"
                yield (filename, result_image, class_label)
            except Exception as e:
                self.logger.exception(f"Failed during text-to-image generation for seed {current_seed}: {e}")

    def _generate_with_lora_img2img(self, params: DictConfig, num_to_generate: int, class_label: str) -> Generator[Tuple, None, None]:
        """Generates new images conditioned on source images and text prompts using a LoRA model."""
        if not _LORA_MODEL_AVAILABLE: return
        class_items = self.dataset.class_to_items_map.get(class_label, [])
        if not class_items:
            self.logger.error(f"No input items found for class '{class_label}'."); return
        input_image_paths = [Path(item[0]) for item in class_items]
        pipeline = self._get_lora_pipeline(params, StableDiffusionImg2ImgPipeline)
        if pipeline is None: return
        prompts = [self.dataset._generate_prompt_for_class(class_label) for _ in range(num_to_generate)]
        image_size = params.generation.get('image_size', 256)
        for i, prompt in enumerate(tqdm(prompts, desc=f"Generating '{class_label}'")):
            current_seed = params.seed + i
            generator = torch.Generator(device=pipeline.device).manual_seed(current_seed)
            init_image_path = input_image_paths[i % len(input_image_paths)]
            original_image = Image.open(init_image_path).convert("RGB")
            resized_init_image = original_image.resize((image_size, image_size), Image.Resampling.LANCZOS)
            try:
                with torch.inference_mode(), torch.autocast(device_type=pipeline.device.type, dtype=getattr(torch, params.compute.dtype)):
                    result_image = pipeline(prompt=prompt, image=resized_init_image, strength=params.generation.strength, guidance_scale=params.generation.guidance_scale, num_inference_steps=params.generation.steps, generator=generator, negative_prompt=params.get("negative_prompt", "")).images[0]
                filename = f"img2img_{class_label.replace(' ', '_').lower()}_{current_seed}.png"
                yield (filename, result_image, resized_init_image, class_label)
            except Exception as e:
                self.logger.exception(f"Failed during image-to-image generation for seed {current_seed}: {e}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
@hydra.main(config_path="config", config_name="gen_images", version_base=None)
def main(cfg: DictConfig):
    
    console_logger = setup_logger("main", level=logging.INFO)
    console_logger.info("Starting sequential offline data augmentation process...")
    if not cfg.get("augmentation_jobs"):
        console_logger.critical("Config error: 'augmentation_jobs' list not found."); return

    original_cwd = Path(get_original_cwd())
    master_seed = cfg.get('seed', random.randint(0, 10_000_000))
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_offset = 0
    
    global_stylegan_path = cfg.get('stylegan3_repo_path')

    for job_entry in cfg.augmentation_jobs:
        if not job_entry.get('enabled', True):
            console_logger.info(f"Job '{job_entry.name}' is disabled. Skipping.")
            continue
        
        console_logger.info(f"--- Starting Job: {job_entry.name} ---")

        # Get source dataset config directly using the key from the job
        source_dataset_name = job_entry.dataset
        if source_dataset_name not in cfg.datasets:
            console_logger.error(f"Dataset '{source_dataset_name}' not found in _datasets.yaml. Skipping job.")
            continue
        source_cfg = cfg.datasets[source_dataset_name]

        # Build output directory using the global base path
        output_dir = original_cwd / cfg.output_base_path / run_timestamp / job_entry.name
        output_dir.mkdir(parents=True, exist_ok=True)
        job_logger = setup_logger(name=f"Job-{job_entry.name}", log_file=output_dir / "generation.log")

        # Prepare a mutable copy of the job config to add runtime parameters
        final_job_cfg_container = OmegaConf.to_container(job_entry, resolve=True)
        final_job_cfg_container['params']['seed'] = master_seed + seed_offset
        if 'stylegan' in job_entry.type and global_stylegan_path:
            final_job_cfg_container['params']['stylegan_repo_path'] = global_stylegan_path
        final_job_cfg = OmegaConf.create(final_job_cfg_container)

        try:
            base_data_path = original_cwd / source_cfg.base_path
            dataset_init_kwargs = {}
            if source_cfg.type == 'json':
                source_path = base_data_path / source_cfg.train_file
                if 'image_folder' in source_cfg and source_cfg.image_folder:
                     dataset_init_kwargs['image_folder'] = source_cfg.image_folder
            elif source_cfg.type == 'directory':
                source_path = base_data_path / source_cfg.train_file
            else:
                raise ValueError(f"Unsupported dataset type in config: '{source_cfg.type}'")
            
            job_logger.info(f"Initializing dataset from: {source_path}")
            if not source_path.exists():
                raise FileNotFoundError(f"Source data path does not exist: {source_path}")
            
            dataset = DiffusionDataset(data_path=str(source_path), **dataset_init_kwargs)
            
            if not dataset.class_names:
                job_logger.warning("No classes found in source dataset. Skipping job."); continue
        except Exception as e:
            console_logger.error(f"Could not load source data for job '{job_entry.name}'. Error: {e}. Skipping."); continue

        try:
            augmenter = DatasetAugmenter(final_job_cfg, dataset=dataset, logger=job_logger, output_dir=output_dir)
            augmenter.run()
            console_logger.info(f"--- Finished Job: {job_entry.name} ---")
        except Exception as e:
            job_logger.exception(f"An unhandled exception occurred in job '{job_entry.name}': {e}")
            console_logger.error(f"Job '{job_entry.name}' failed. See logs in {output_dir} for details.")
            
        num_classes = len(dataset.class_names)
        images_generated_in_job = num_classes * job_entry.num_per_class
        seed_offset += images_generated_in_job

    console_logger.info("All augmentation jobs have completed.")

if __name__ == "__main__":
    main()