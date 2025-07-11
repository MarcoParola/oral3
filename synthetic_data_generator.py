# =============================================================================
# IMPORTS
# =============================================================================

import logging
import random
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

import hydra
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

# --- Local Application Imports ---
try:
    from src.data.stable_diffusion.dataset import DiffusionDataset
except ImportError as e:
    print(f"Error: Could not import DiffusionDataset. Ensure 'src' is in the Python path.")
    DiffusionDataset = None

try:
    from src.models.stable_diffusion import LoraDiffusionModel
    _LORA_MODEL_AVAILABLE = True
except ImportError:
    print("WARNING: LoraDiffusionModel not found. 'lora_generate' will be unavailable.")
    LoraDiffusionModel = None

# =============================================================================
# SCRIPT SETUP
# =============================================================================

def setup_logger(name: str, log_file: Path = None, level=logging.INFO) -> logging.Logger:
    """
    Configures and returns a logger instance.

    If a log file path is provided, the logger writes to that file. Otherwise, it
    writes to the console. This allows for creating dedicated file loggers for
    each thread and a separate console logger for the main thread.

    Args:
        name: The name of the logger.
        log_file: Optional path to a file for log output.
        level: The logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent logs from bubbling up to the root logger

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler(log_file) if log_file else logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

# --- Global Threading Primitives ---
# Ensures only one thread uses the GPU for model loading or inference at a time.
GPU_LOCK = threading.Lock()
# Ensures thread-safe access to the shared pipeline cache.
CACHE_LOCK = threading.Lock()
# Caches loaded LoRA pipelines to avoid redundant disk I/O and VRAM usage.
_LORA_PIPELINE_CACHE: Dict[str, StableDiffusionPipeline] = {}


# =============================================================================
# AUGMENTATION WORKFLOW CLASS
# =============================================================================

class DatasetAugmenter:
    """
    Encapsulates the augmentation workflow for a single, fully configured job.
    This class is instantiated and run within a dedicated thread.
    """
    def __init__(self, job_cfg: DictConfig, dataset: DiffusionDataset, logger: logging.Logger, output_dir: Path):
        """
        Initializes the augmenter with pre-resolved configurations.

        Args:
            job_cfg: The fully resolved configuration for this specific job.
            dataset: An instantiated DiffusionDataset for the required source data.
            logger: The logger instance dedicated to this job's thread.
            output_dir: The final output directory for this job's generated data.
        """
        self.job_cfg = job_cfg
        self.dataset = dataset
        self.logger = logger
        self.output_dir = output_dir

        class_names = self.dataset.get_class_names()
        if not class_names:
            raise ValueError("No classes were found by the source DiffusionDataset.")
        
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.logger.debug(f"Class-to-index mapping: {self.class_to_idx}")

        num_to_gen_per_class = self.job_cfg.get('num_per_class', 10)
        self.generation_plan = {cls: num_to_gen_per_class for cls in class_names}
        self.logger.info(f"Generation plan created: {self.generation_plan}")

    def run(self):
        """Executes the full pipeline for this job: generate images, then save them."""
        self.logger.info("Starting image generation phase...")
        generated_images = self._generate_images()

        if not generated_images:
            self.logger.warning("No images were generated. Skipping save phase."); return

        self.logger.info("Starting dataset materialization phase...")
        self._save_images_to_directory(generated_images)

    def _generate_images(self) -> List[Tuple[str, Image.Image, str]]:
        """
        Dispatches to the correct generation engine and manages seed progression
        across all classes within this job.
        """
        engine_map = {'lora_generate': self._generate_with_lora, 'gan': self._generate_with_gan}
        synthesis_engine = engine_map.get(self.job_cfg.type)
        if not synthesis_engine:
            self.logger.error(f"Invalid synthesis engine '{self.job_cfg.type}' specified."); return []
        
        all_synthesized_samples = []
        job_start_seed = self.job_cfg.params.seed
        class_seed_offset = 0
        
        for class_label, num_to_generate in self.generation_plan.items():
            if num_to_generate > 0:
                self.logger.info(f"Generating {num_to_generate} images for class '{class_label}'...")
                
                # Create a mutable copy of parameters for this specific call
                current_params = self.job_cfg.params.copy()
                
                # Calculate and assign a unique starting seed for this class
                class_start_seed = job_start_seed + class_seed_offset
                current_params.seed = class_start_seed
                self.logger.debug(f"Assigning starting seed {class_start_seed} for class '{class_label}'.")
                
                # Call the selected generation engine
                generated_tuples = synthesis_engine(params=current_params, num_to_generate=num_to_generate, class_label=class_label)
                all_synthesized_samples.extend(generated_tuples)
                
                # Increment the offset to ensure the next class uses a new block of seeds
                class_seed_offset += num_to_generate
                
        return all_synthesized_samples

    def _save_images_to_directory(self, images: List[Tuple[str, Image.Image, str]]):
        """Saves generated images to the output directory in ImageFolder format."""
        self.logger.info(f"Saving {len(images)} images to: {self.output_dir}")
        train_output_dir = self.output_dir / "train"
        
        # To preserve the log file, only clear the 'train' subdirectory if it exists
        if train_output_dir.exists():
            self.logger.warning(f"Train directory already exists. Clearing contents: {train_output_dir}")
            shutil.rmtree(train_output_dir)
        train_output_dir.mkdir(parents=True, exist_ok=True)

        for filename, pil_img, class_label in tqdm(images, desc=f"Saving to {self.output_dir.name}", unit="image"):
            class_output_dir = train_output_dir / class_label
            class_output_dir.mkdir(exist_ok=True)
            if pil_img.mode != 'RGB': 
                pil_img = pil_img.convert('RGB')
            jpg_filename = Path(filename).with_suffix('.jpg').name
            pil_img.save(class_output_dir / jpg_filename, format='JPEG', quality=95)
        self.logger.info(f"Successfully created dataset at {self.output_dir}")

    def _generate_with_lora(self, params: DictConfig, num_to_generate: int, class_label: str) -> List[Tuple[str, Image.Image, str]]:
        """Generates images using a fine-tuned LoRA/Diffusion model."""
        if not _LORA_MODEL_AVAILABLE: self.logger.error("LoraDiffusionModel not available."); return []
        
        ckpt_path = Path(params.ckpt_path).resolve()
        # Thread-safe access to the pipeline cache
        with CACHE_LOCK:
            pipeline = _LORA_PIPELINE_CACHE.get(str(ckpt_path))
        
        if pipeline is None:
            self.logger.info(f"Loading LoRA model (first time): {ckpt_path.name}")
            try:
                # Acquire GPU lock for VRAM-intensive model loading
                with GPU_LOCK:
                    self.logger.debug("Acquired GPU lock for model loading.")
                    device = params.compute.get("device", "cuda")
                    if device == "cuda" and not torch.cuda.is_available():
                        device = "cpu"; self.logger.warning("CUDA not available, falling back to CPU.")
                    pl_model = LoraDiffusionModel.load_from_checkpoint(str(ckpt_path), map_location=device, strict=False)
                    pl_model.eval()
                    scheduler = DPMSolverMultistepScheduler.from_pretrained(params.base_model_name, subfolder="scheduler")
                    pipeline = StableDiffusionPipeline(vae=pl_model.vae, text_encoder=pl_model.text_encoder, tokenizer=pl_model.tokenizer, unet=pl_model.unet, scheduler=scheduler, safety_checker=None, feature_extractor=None).to(device)
                # Add the loaded pipeline to the cache
                with CACHE_LOCK:
                    _LORA_PIPELINE_CACHE[str(ckpt_path)] = pipeline
                self.logger.info(f"LoRA pipeline for '{ckpt_path.name}' loaded and cached.")
            except Exception as e:
                self.logger.exception(f"Failed to load LoRA checkpoint: {e}"); return []
        else:
            self.logger.debug(f"Found cached LoRA pipeline for '{ckpt_path.name}'.")

        synthesized_samples = []
        master_seed = params.seed
        use_variation = params.get("use_prompt_variation", False)
        # Generate prompts using the logic from the Dataset class
        prompts = [self.dataset._generate_prompt_for_class(class_label, use_variation=use_variation) for _ in range(num_to_generate)]

        # Acquire GPU lock for the inference loop
        with GPU_LOCK:
            self.logger.debug(f"Acquired GPU lock for image generation (class: {class_label}).")
            for i, prompt in enumerate(tqdm(prompts, desc=f"Generating '{class_label}'")):
                current_seed = master_seed + i
                generator = torch.Generator(device=pipeline.device).manual_seed(current_seed)
                self.logger.debug(f"Generating image {i+1}/{num_to_generate} with seed {current_seed} and prompt: '{prompt}'")
                try:
                    with torch.inference_mode(), torch.autocast(device_type=pipeline.device.type, dtype=getattr(torch, params.compute.dtype)):
                        result_image = pipeline(prompt=prompt, negative_prompt=params.get("negative_prompt", ""), num_inference_steps=params.generation.steps, guidance_scale=params.generation.guidance_scale, generator=generator).images[0]
                    filename = f"lora_{class_label.replace(' ', '_').lower()}_{current_seed}.png"
                    synthesized_samples.append((filename, result_image, class_label))
                except Exception as e:
                    self.logger.exception(f"Failed during image generation for seed {current_seed}: {e}")
        self.logger.debug(f"Released GPU lock after generating for class {class_label}.")
        return synthesized_samples

    def _generate_with_gan(self, params: DictConfig, num_to_generate: int, class_label: str) -> List[Tuple[str, Image.Image, str]]:
        """Generates images by calling an external StyleGAN3 script."""
        if not all(k in params for k in ['stylegan_repo_path', 'network_pkl']):
            self.logger.error("StyleGAN config missing 'stylegan_repo_path' or 'network_pkl'."); return []
        
        repo_path = Path(params.stylegan_repo_path)
        gen_script_name = 'gen_images.py'
        if not (repo_path / gen_script_name).is_file():
            self.logger.error(f"StyleGAN script '{gen_script_name}' not found in: {repo_path}"); return []

        network_pkl = Path(params.network_pkl).resolve()
        class_idx = self.class_to_idx.get(class_label)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            master_seed = params.seed
            seeds = ','.join(map(str, range(master_seed, master_seed + num_to_generate)))
            
            command = [sys.executable, gen_script_name, f'--network={network_pkl}', f'--outdir={temp_path}', f'--seeds={seeds}', f'--class={class_idx}']
            if (psi := params.get('truncation_psi')) is not None: command.append(f'--trunc={psi}')
            
            # Acquire GPU lock to run the StyleGAN subprocess
            with GPU_LOCK:
                self.logger.debug(f"Acquired GPU lock for GAN generation (class: {class_label}).")
                try:
                    subprocess.run(command, check=True, capture_output=True, text=True, cwd=repo_path)
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"StyleGAN script failed. Stderr:\n{e.stderr}"); return []
            self.logger.debug(f"Released GPU lock after generating for class {class_label}.")

            synthesized_samples = []
            for img_path in sorted(list(temp_path.glob('*.png'))):
                with Image.open(img_path) as pil_img: pil_img.load()
                # Extract the seed from the original filename for consistency
                original_seed = Path(img_path).stem.split('-')[-1]
                filename = f"stylegan_{class_label.replace(' ', '_').lower()}_{original_seed}.png"
                synthesized_samples.append((filename, pil_img, class_label))
        return synthesized_samples

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_job_worker(job_cfg: DictConfig, source_cfg: DictConfig, cwd: Path, run_timestamp: str):
    """
    The target function for each thread, responsible for running one augmentation job.
    """
    # 1. Construct the final output path for this job.
    model_profile = job_cfg.model_profile
    output_dir = cwd / job_cfg.output_base_path / run_timestamp / model_profile
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Setup a dedicated file logger for this specific thread.
    log_level = getattr(logging, job_cfg.get('log_level', 'INFO').upper(), logging.INFO)
    logger = setup_logger(
        name=threading.current_thread().name,
        log_file=output_dir / "generation.log",
        level=log_level
    )
    
    logger.info(f"Thread worker started for job: {job_cfg.name}")
    try:
        # 3. Instantiate the source dataset and the augmenter, then run the job.
        base_source_path = cwd / source_cfg.base_path / source_cfg.name / source_cfg.source_type
        source_path = base_source_path / 'train' if (base_source_path / 'train').is_dir() else base_source_path / 'train.json'
        dataset = DiffusionDataset(data_path=str(source_path))

        augmenter = DatasetAugmenter(job_cfg, dataset=dataset, logger=logger, output_dir=output_dir)
        augmenter.run()
        logger.info(f"Job '{job_cfg.name}' finished successfully.")
    except Exception as e:
        logger.exception(f"An unhandled exception occurred in job '{job_cfg.name}': {e}")

@hydra.main(config_path="config", config_name="dataset_creation", version_base=None)
def main(cfg: DictConfig):
    """
    Main entry point that resolves configurations, manages seeds, and dispatches
    jobs to worker threads.
    """
    console_logger = setup_logger("main", level=logging.INFO)
    console_logger.info("Starting multi-threaded offline data augmentation process...")
    if not cfg.get("augmentation_jobs"):
        console_logger.critical("Config error: 'augmentation_jobs' list not found."); return

    original_cwd = Path(get_original_cwd())
    
    # --- Setup for the entire run ---
    master_seed = cfg.get('seed', random.randint(0, 10_000_000))
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    console_logger.info(f"Run started with master seed: {master_seed} and timestamp: {run_timestamp}")
    
    threads_to_run = []
    seed_offset = 0

    # 1. Resolve all configurations sequentially to avoid race conditions.
    console_logger.info("Resolving configurations for all jobs...")
    for job_entry in cfg.augmentation_jobs:
        if not job_entry.get('enabled', True):
            console_logger.info(f"Job '{job_entry.name}' is disabled. Skipping."); continue
        
        # Look up profiles from the libraries
        model_profile_name = job_entry.model_profile
        model_cfg = cfg.model_library.get(model_profile_name)
        source_dataset_name = model_cfg.source_dataset_name
        source_cfg = cfg.source_dataset_library.get(source_dataset_name)

        # Pre-calculate number of images to correctly offset seeds for the next job
        base_source_path = original_cwd / source_cfg.base_path / source_cfg.name / source_cfg.source_type
        source_path = base_source_path / 'train' if (base_source_path / 'train').is_dir() else base_source_path / 'train.json'
        try:
            num_classes = len(DiffusionDataset(data_path=str(source_path)).get_class_names())
        except FileNotFoundError:
            console_logger.error(f"Source data not found for job '{job_entry.name}' at path: {source_path}. Skipping job."); continue
        images_to_generate_in_job = num_classes * job_entry.num_per_class
        
        # Merge configurations and inject the calculated seed
        final_dict = {**OmegaConf.to_container(model_cfg, resolve=True), **OmegaConf.to_container(job_entry, resolve=True)}
        final_dict['output_base_path'] = cfg.output_base_path
        job_seed = master_seed + seed_offset
        final_dict.setdefault('params', {})['seed'] = job_seed
        
        threads_to_run.append({'job_cfg': OmegaConf.create(final_dict), 'source_cfg': source_cfg})
        seed_offset += images_to_generate_in_job

    # 2. Launch each resolved job in a separate thread.
    console_logger.info(f"Launching {len(threads_to_run)} jobs. See individual log files for details.")
    threads = []
    for job_info in threads_to_run:
        job_name = job_info['job_cfg'].name
        thread = threading.Thread(
            target=run_job_worker,
            args=(job_info['job_cfg'], job_info['source_cfg'], original_cwd, run_timestamp),
            name=job_name
        )
        threads.append(thread)
        thread.start()
        console_logger.info(f"--> Dispatched job: '{job_name}'")

    # 3. Wait for all threads to complete.
    for thread in threads:
        thread.join()
        console_logger.info(f"<-- Completed job: '{thread.name}'")
            
    console_logger.info("All augmentation jobs have completed.")

if __name__ == "__main__":
    main()