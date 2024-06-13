import argparse
import json
import random
import os
from pathlib import Path

def create_coco(images, annotations, categories):
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

def load_dataset(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

def split_dataset(images_dataset, contrastive_dataset, train_perc, val_perc):
    # Carica le informazioni relative ad annotazioni e categorie dal file dataset.json
    dataset = load_dataset(images_dataset)
    annotations = dataset.get("annotations", [])
    categories = dataset.get("categories", [])

    # Carica le informazioni relative alle immagini dal file contrastive_dataset.json
    with open(contrastive_dataset, 'r') as file:
        contrastive_data = json.load(file)

    images = contrastive_data.get("images", [])

    images_by_category = {}
    for image in images:
        category_id = image["category_id"]
        if category_id not in images_by_category:
            images_by_category[category_id] = []
        images_by_category[category_id].append(image)

    # Crea una lista piatta di annotazioni
    annotations_flat = []
    for annotation in annotations:
        if annotation["image_id"] in {image["id"] for image in images}:
            annotations_flat.append(annotation)

    train_set = []
    val_set = []
    test_set = []

    category_names = {
        0: "neoplastic",
        1: "aphthous",
        2: "traumatic"
    }

    for category_id, images in images_by_category.items():
        random.shuffle(images)
        total_images = len(images)
        train_size = int(total_images * train_perc)
        val_size = int(total_images * val_perc)
        train_set.extend(images[:train_size])
        val_set.extend(images[train_size:train_size + val_size])
        test_set.extend(images[train_size + val_size:])

        print(f"Category: {category_names[category_id]}, Images: {total_images}")

    return train_set, val_set, test_set, annotations_flat

def main():
    parser = argparse.ArgumentParser(
        description="Script for splitting a contrastive dataset into train, validation, and test sets while maintaining category distribution.")
    parser.add_argument("--folder", type=str, required=True, help="Folder containing the dataset JSON file.")
    args = parser.parse_args()

    images_dataset = os.path.join(args.folder, "dataset.json")
    contrastive_dataset = os.path.join(args.folder, "contrastive_dataset_min.json")

    # Esegui la suddivisione
    random.seed(42)  # Imposta il seme del generatore di numeri casuali per riproducibilità
    train_set, val_set, test_set, image_annotations = split_dataset(images_dataset, contrastive_dataset, 0.7, 0.15)

    # Carica le categorie dal file dataset.json
    dataset_categories = load_dataset(images_dataset).get("categories", [])

    # Crea i file JSON di output includendo annotazioni e categorie
    output_folder = Path(args.folder)
    save_json(create_coco(train_set, image_annotations, dataset_categories), output_folder / "contrastive_train_min.json")
    save_json(create_coco(val_set, image_annotations, dataset_categories), output_folder / "contrastive_val_min.json")
    save_json(create_coco(test_set, image_annotations, dataset_categories), output_folder / "contrastive_test_min.json")

    print("OK!")

if __name__ == "__main__":
    main()
