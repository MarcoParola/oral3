# split_dataset.py

import argparse
import logging
import random
import shutil
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

# --- Setup basic logging ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def copy_files(files: List[Path], destination_dir: Path):
    """
    Copies a list of files to a destination directory, showing a progress bar.

    Args:
        files: A list of Path objects representing the files to copy.
        destination_dir: The Path object for the directory to copy files into.
    """
    if not files:
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    for file_path in tqdm(files, desc=f"Copying to {destination_dir.name}", unit="file"):
        shutil.copy(file_path, destination_dir / file_path.name)

def split_image_dataset(
    source_dir: Path,
    output_dir: Path,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    clean_output: bool = False
):
    """
    Splits an ImageFolder-style dataset into training, validation, and test sets.

    Args:
        source_dir: The path to the source directory, which contains subdirectories
                    for each class.
        output_dir: The path to the output directory where the split dataset
                    will be saved.
        ratios: A tuple of three floats representing the split ratios for
                (train, validation, test). Must sum to 1.0.
        seed: A random seed for shuffling to ensure reproducible splits.
        clean_output: If True, deletes the output directory if it already exists.
    """
    # --- 1. Validation and Setup ---
    if not source_dir.is_dir():
        logger.error(f"Source directory not found: {source_dir}")
        return

    if round(sum(ratios), 5) != 1.0:
        logger.error(f"Split ratios must sum to 1.0. Got: {ratios} (sum={sum(ratios)})")
        return

    if output_dir.exists() and clean_output:
        logger.warning(f"Output directory {output_dir} already exists. Cleaning it...")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output will be saved to: {output_dir}")

    # Set the seed for reproducibility
    random.seed(seed)

    # --- 2. Discover classes and process each one ---
    class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        logger.error(f"No class subdirectories found in {source_dir}.")
        return

    logger.info(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")

    for class_dir in class_dirs:
        class_name = class_dir.name
        logger.info(f"--- Processing class: {class_name} ---")

        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
        image_files = [p for p in class_dir.glob('*') if p.suffix.lower() in image_extensions]
        
        if not image_files:
            logger.warning(f"No images found for class '{class_name}'. Skipping.")
            continue

        # Shuffle the list of files for a random split
        random.shuffle(image_files)
        
        # Calculate split indices
        num_images = len(image_files)
        train_split_idx = int(num_images * ratios[0])
        val_split_idx = train_split_idx + int(num_images * ratios[1])
        
        # Slice the file list into train, val, and test sets
        train_files = image_files[:train_split_idx]
        val_files = image_files[train_split_idx:val_split_idx]
        test_files = image_files[val_split_idx:]
        
        logger.info(f"Splitting {num_images} images into: "
                    f"{len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

        # --- 3. Create destination directories and copy files ---
        copy_files(train_files, output_dir / "train" / class_name)
        copy_files(val_files, output_dir / "val" / class_name)
        copy_files(test_files, output_dir / "test" / class_name)

    logger.info("\nDataset splitting process completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split an ImageFolder dataset into train, validation, and test sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to the source dataset directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where the split dataset will be saved."
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        metavar=('TRAIN', 'VAL', 'TEST'),
        help="A tuple of three floats for train, validation, and test split ratios."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling."
    )
    parser.add_argument(
        "--clean_output",
        action='store_true',
        help="If set, deletes the output directory before starting if it already exists."
    )
    
    args = parser.parse_args()

    split_image_dataset(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        ratios=tuple(args.ratios),
        seed=args.seed,
        clean_output=args.clean_output
    )