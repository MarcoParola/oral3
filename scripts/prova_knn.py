import json
import random
import csv

# Percorso del file di input
input_file = 'data/dataset.json'
# Percorso del file di output
output_file = 'data/knn_dataset.json'
# Percorso del file CSV
csv_file = 'data/anchor_dataset.csv'

# Carica i dati dal file JSON
with open(input_file, 'r') as f:
    data = json.load(f)

# Carica i valori di "id_casi" dal file CSV
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    id_casi = [row['id_casi'] for row in reader]

# Crea un dizionario image_id -> category_id
image_to_category = {ann['image_id']: ann['category_id'] for ann in data['annotations']}

# Filtra le immagini in base ai valori di "file_name" nel CSV
selected_images = [item for item in data['images'] if item['file_name'] in id_casi]

# Assegna category_id a ciascuna immagine selezionata
for image in selected_images:
    image['category_id'] = image_to_category.get(image['id'], None)

# Raggruppa le immagini selezionate per categoria
selected_category_0 = [item for item in selected_images if item.get('category_id') == 1]
selected_category_1 = [item for item in selected_images if item.get('category_id') == 2]
selected_category_2 = [item for item in selected_images if item.get('category_id') == 3]

print(f"Category 0: {len(selected_category_0)} items")
print(f"Category 1: {len(selected_category_1)} items")
print(f"Category 2: {len(selected_category_2)} items")

# Definisci il numero desiderato di elementi per ciascuna categoria
desired_count_neoplastic = 179
desired_count_aphthous = 189
desired_count_traumatic = 207

# Completa gli elementi mancanti per ciascuna categoria
def complete_category(selected_category, all_category, count):
    if len(selected_category) < count:
        available_to_add = [item for item in all_category if item not in selected_category]
        if len(available_to_add) >= (count - len(selected_category)):
            additional_items = random.sample(available_to_add, count - len(selected_category))
        else:
            additional_items = available_to_add
        selected_category.extend(additional_items)
    return selected_category

# Ottieni tutte le immagini per ciascuna categoria
category_0 = [item for item in data['images'] if image_to_category.get(item['id']) == 1]
category_1 = [item for item in data['images'] if image_to_category.get(item['id']) == 2]
category_2 = [item for item in data['images'] if image_to_category.get(item['id']) == 3]

# Completa ciascuna categoria
selected_category_0 = complete_category(selected_category_0, category_0, desired_count_neoplastic)
selected_category_1 = complete_category(selected_category_1, category_1, desired_count_aphthous)
selected_category_2 = complete_category(selected_category_2, category_2, desired_count_traumatic)

print(f"Category 0 neoplastic: {len(selected_category_0)} items")
print(f"Category 1 aphthous: {len(selected_category_1)} items")
print(f"Category 2 traumatic: {len(selected_category_2)} items")

# Combina i dati selezionati
selected_data = selected_category_0 + selected_category_1 + selected_category_2

# Salva i dati selezionati nel nuovo file JSON
with open(output_file, 'w') as f:
    json.dump({'images': selected_data}, f, indent=4)