import argparse
import json
import random
import os
import csv

def most_frequent(arr): 
    return max(set(arr), key=arr.count) 

def create_coco(images):
    global images_ids
    global dataset
    coco_images = []
    coco_annotations = []
    
    for image in images:
        coco_images.append(images_ids[image["id"]])
        for annotation in image["annotations"]:
            coco_annotations.append(annotation)
    
    return dict(
        images=coco_images,
        annotations=coco_annotations,
        categories=dataset["categories"]
    )

def load_csv_ids(file_path):
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        return [row['id_casi'] for row in reader]

def ensure_unique_filenames(images):
    seen_filenames = set()
    unique_images = []
    for image in images:
        filename = images_ids[image["id"]]["file_name"]
        if filename not in seen_filenames:
            seen_filenames.add(filename)
            unique_images.append(image)
    return unique_images

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True)
parser.add_argument("--train-perc", type=float, default=0.7)
parser.add_argument("--val-perc", type=float, default=0.15)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)

dataset = json.load(open(os.path.join(args.folder, "dataset.json"), "r"))

category_buckets = dict()
category_names = dict()
images_ids = dict()
images = dict()

for image in dataset["images"]:
    images_ids[image["id"]] = image

for category in dataset["categories"]:
    category_names[category["id"]] = category["name"]

for annotation in dataset["annotations"]:
    if annotation["image_id"] not in images:
        images[annotation["image_id"]] = dict(annotations=[], c=None, id=annotation["image_id"])
    images[annotation["image_id"]]["annotations"].append(annotation)

for id, image in images.items():
    categories = list(map(lambda a: a["category_id"], image["annotations"]))
    image["c"] = most_frequent(categories)

for id, image in images.items():
    if image["c"] not in category_buckets:
        category_buckets[image["c"]] = []
    category_buckets[image["c"]].append(image)

# Carica gli ID delle immagini da anchor_dataset.csv
anchor_csv_path = os.path.join(args.folder, "anchor_dataset.csv")
anchor_ids = load_csv_ids(anchor_csv_path)
print(f"IDs from CSV: {anchor_ids}")
print(f"len(anchor_ids): {len(anchor_ids)}")

test_images = []
train_images = []
val_images = []

# Forza la selezione delle immagini per il test set utilizzando gli ID di anchor_dataset.csv
for category_id, images in category_buckets.items():
    random.shuffle(images)
    test_images_for_category = [img for img in images if images_ids[img["id"]]["file_name"] in anchor_ids]
    remaining_images_for_category = [img for img in images if images_ids[img["id"]]["file_name"] not in anchor_ids]
    
    # Assicurati che i file_name nel test set siano unici
    test_images_for_category = ensure_unique_filenames(test_images_for_category)

    # Completa il test set se necessario con immagini prese casualmente
    required_test_size = int(len(images) * (1 - args.train_perc - args.val_perc)) - len(test_images_for_category)
    if required_test_size > 0:
        test_images_for_category.extend(ensure_unique_filenames(remaining_images_for_category[:required_test_size]))
        remaining_images_for_category = remaining_images_for_category[required_test_size:]
    
    test_images.extend(test_images_for_category)

    # Calcola le dimensioni per train e val
    total_images = len(images)
    train_size = int(total_images * args.train_perc)
    val_size = int(total_images * args.val_perc)

    train_images_for_category = remaining_images_for_category[:train_size]
    val_images_for_category = remaining_images_for_category[train_size:train_size + val_size]

    train_images.extend(train_images_for_category)
    val_images.extend(val_images_for_category)

    print(f"Category: {category_id}, Train: {len(train_images_for_category)}, Val: {len(val_images_for_category)}, Test: {len(test_images_for_category)}")

# Assicurati che i file_name nel test set siano unici
test_images = ensure_unique_filenames(test_images)

json.dump(create_coco(train_images), open(os.path.join(args.folder, "knn_train.json"), "w"), indent=2)
json.dump(create_coco(val_images), open(os.path.join(args.folder, "knn_val.json"), "w"), indent=2)
json.dump(create_coco(test_images), open(os.path.join(args.folder, "knn_test.json"), "w"), indent=2)
print("OK!")
