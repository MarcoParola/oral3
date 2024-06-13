import argparse
import json

def compute_stats(json_data):
    category_count = {0: 0, 1: 0, 2: 0}  # Inizializza il conteggio delle categorie

    for image in json_data["images"]:
        category_id = image["category_id"]
        category_count[category_id] += 1

    category_names = {
        0: "neoplastic",
        1: "aphthous",
        2: "traumatic"
    }

    for category_id, count in category_count.items():
        category_name = category_names.get(category_id, "Unknown")
        print(f"{category_name}: {count}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset JSON file.")
    args = parser.parse_args()
    dataset = json.load(open(args.dataset, "r"))
    compute_stats(dataset)
