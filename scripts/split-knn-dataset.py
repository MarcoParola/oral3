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

def split_dataset(images_dataset, knn_dataset, train_perc, val_perc):
    # Carica le informazioni relative ad annotazioni e categorie dal file dataset.json
    dataset = load_dataset(images_dataset)
    annotations = dataset.get("annotations", [])
    categories = dataset.get("categories", [])

    # Carica le informazioni relative alle immagini dal file knn_dataset.json
    with open(knn_dataset, 'r') as file:
        knn_data = json.load(file)

    images = knn_data.get("images", [])

    # Crea un dizionario image_id -> category_id dalle annotazioni
    image_to_category = {ann['image_id']: ann['category_id'] for ann in annotations}

    # Aggiungi il category_id a ogni immagine
    for image in images:
        image_id = image["id"]
        image["category_id"] = image_to_category.get(image_id)

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
        1: "neoplastic",
        2: "aphthous",
        3: "traumatic"
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
        description="Script for splitting a knn dataset into train, validation, and test sets while maintaining category distribution.")
    parser.add_argument("--folder", type=str, required=True, help="Folder containing the dataset JSON file.")
    args = parser.parse_args()

    images_dataset = os.path.join(args.folder, "dataset.json")
    knn_dataset = os.path.join(args.folder, "knn_dataset.json")

    # Esegui la suddivisione
    random.seed(42)  # Imposta il seme del generatore di numeri casuali per riproducibilità
    train_set, val_set, test_set, image_annotations = split_dataset(images_dataset, knn_dataset, 0.7, 0.15)

    # Carica le categorie dal file dataset.json
    dataset_categories = load_dataset(images_dataset).get("categories", [])

    # Crea i file JSON di output includendo annotazioni e categorie
    output_folder = Path(args.folder)
    save_json(create_coco(train_set, image_annotations, dataset_categories), output_folder / "knn_train.json")
    save_json(create_coco(val_set, image_annotations, dataset_categories), output_folder / "knn_val.json")
    save_json(create_coco(test_set, image_annotations, dataset_categories), output_folder / "knn_test.json")

    # stampa le informazioni sui set di dati specificando il numero di immagini per categoria in ciascun set
    print("Dataset split:")
    print(f"Train set: {len(train_set)} images")
    print(f"Validation set: {len(val_set)} images")
    print(f"Test set: {len(test_set)} images")

    print("OK!")

if __name__ == "__main__":
    main()
