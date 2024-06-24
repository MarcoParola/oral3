import torch
import torchvision.transforms as transforms
import os
import json
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

class OralContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

        with open(images, "r") as f:
            self.dataset = json.load(f)

        with open("data/dataset.json", "r") as f:
            self.contrastive_data = json.load(f)

        self.annotations = dict()
        for annotation in self.dataset["annotations"]:
            self.annotations[annotation["image_id"]] = annotation

        self.categories = dict()
        for i, category in enumerate(self.dataset["categories"]):
            self.categories[category["id"]] = i

    def __len__(self):
        return len(self.dataset["images"])

    def __getitem__(self, idx):
        image = self.dataset["images"][idx] 
        annotation = self.annotations[image["id"]]
        
        image_id = image["id"]
        image_name = image["file_name"]
        if "positive" not in image or "negative" not in image:
            positive = image["file_name"] # usato per il test set
            negative = image["file_name"] # usato per il test set
        else:
            positive = image["positive"]  # Ottieni il nome del file positivo
            negative = image["negative"]  # Ottieni il nome del file negativo

        # ottieni l'immagine positiva
        positive_image = None
        for img in self.contrastive_data["images"]:
            if img["file_name"] == positive:
                positive_image = img
                break
        
        # Controlla se l'immagine positiva è stata trovata
        if positive_image is None:
            positive_image = image # usato per il test set
        #    raise ValueError(f"No positive image found for {image_name}")

        # ottieni l'immagine negativa
        negative_image = None
        for img in self.contrastive_data["images"]:
            if img["file_name"] == negative:
                negative_image = img
                break
        
        # Controlla se l'immagine negativa è stata trovata
        if negative_image is None:
            negative_image = image # usato per il test set
        #    raise ValueError(f"No negative image found for {image_name}")

        image_path = os.path.join(os.path.dirname(self.images), "oral1", image["file_name"])
        image = Image.open(image_path).convert("RGB")

        positive_image_path = os.path.join(os.path.dirname(self.images), "oral1", positive_image["file_name"])
        positive_image = Image.open(positive_image_path).convert("RGB")

        negative_image_path = os.path.join(os.path.dirname(self.images), "oral1", negative_image["file_name"])
        negative_image = Image.open(negative_image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        category = self.categories[annotation["category_id"]]

        return image, category, image_id, image_name, positive, negative, positive_image, negative_image
