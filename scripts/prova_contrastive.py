import json
import random

# Percorso del file di input
#input_file = 'data/contrastive_dataset.json'
input_file = 'data/contrastive_dataset_max.json'
# Percorso del file di output
output_file = 'data/contrastive_dataset_min.json'

# Carica i dati dal file JSON
with open(input_file, 'r') as f:
    data = json.load(f)

# Filtra gli elementi in base ai category_id
category_0 = [item for item in data['images'] if item['category_id'] == 0]
category_1 = [item for item in data['images'] if item['category_id'] == 1]
category_2 = [item for item in data['images'] if item['category_id'] == 2]

# Seleziona casualmente 30 elementi per ciascuna categoria
selected_category_0 = random.sample(category_0, 150)
selected_category_1 = random.sample(category_1, 150)
selected_category_2 = random.sample(category_2, 150)

# Combina i dati selezionati
selected_data = selected_category_0 + selected_category_1 + selected_category_2

# Salva i dati selezionati nel nuovo file JSON
with open(output_file, 'w') as f:
    json.dump({'images': selected_data}, f, indent=4)

print(f"Created {output_file} with same items from each category (0, 1, 2).")
