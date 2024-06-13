import json
import csv

# Percorso del file di input
input_file = 'data/dataset.json'
# Percorso del file di output
output_file = 'data/fsl_test.json'
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

# Filtra le immagini in base ai valori di "file_name" non nel file CSV
selected_images = [image for image in data['images'] if image['file_name'] not in id_casi]

# Assegna category_id a ciascuna immagine selezionata
for image in selected_images:
    image['category_id'] = image_to_category.get(image['id'], None)

# Definisci il numero desiderato di elementi per ciascuna categoria
desired_count = 10

# Raggruppa le immagini selezionate per categoria
categories = {1: [], 2: [], 3: []}
for image in selected_images:
    if image['category_id'] in categories:
        categories[image['category_id']].append(image)

# Limita il numero di immagini per categoria al numero desiderato
final_selection = []
for cat_id, images in categories.items():
    final_selection.extend(images[:desired_count])

# Filtra le annotazioni basate sulle immagini selezionate
selected_image_ids = {image['id'] for image in final_selection}
selected_annotations = [ann for ann in data['annotations'] if ann['image_id'] in selected_image_ids]

# Ottieni tutte le categorie dal file JSON originale
all_categories = data['categories']

# Salva i dati selezionati nel nuovo file JSON
with open(output_file, 'w') as f:
    json.dump({
        'images': final_selection,
        'annotations': selected_annotations,
        'categories': all_categories
    }, f, indent=4)

print(f"Immagini selezionate: {len(final_selection)}")