import torch
import torchvision.transforms as transforms
import os
import json
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel


class OralFewShotDataset(torch.utils.data.Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

        with open(images, "r") as f:
            self.dataset = json.load(f)

        self.annotations = dict()
        for annotation in self.dataset["annotations"]:
            self.annotations[annotation["image_id"]] = annotation

        self.categories = dict()
        for i, category in enumerate(self.dataset["categories"]):
            self.categories[category["id"]] = i
        
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
    def __len__(self):
        return len(self.dataset["images"])
    
    def __getitem__(self, idx):
        # ottieni l'immagine, label e descrizione
        image = self.dataset["images"][idx]
        annotation = self.annotations[image["id"]]
        image_id = image["id"]
        image_name = image["file_name"]
        rule = image.get("rule", None)


        image_path = os.path.join(os.path.dirname(self.images), "oral1", image["file_name"])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        category = self.categories[annotation["category_id"]]

        # ottieni l'embedding dalla rule usando DistilBERT se presente
        if rule is None:
            rule_embedding = torch.zeros(768)
        else:
            inputs = self.tokenizer(rule, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model(**inputs)
            rule_embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)

        return image, category, image_id, image_name, rule_embedding
    
if __name__ == "__main__":
    
    dataset = OralFewShotDataset(
        "data/few_shot_dataset.json",
        transform=transforms.Compose([
            transforms.Resize((512, 512), antialias=True),
            transforms.ToTensor()
        ])
    )    
    
    image, category, image_id, image_name, rule_embedding = dataset.__getitem__(28)
    
    plt.imshow(image.permute(1, 2, 0))
    plt.show()

    # Stampa le informazioni
    print("category", category)
    print("image_id", image_id)
    print("image_name", image_name)
    print("rule_embedding", rule_embedding)


    
