import torch
import torchvision.transforms as transforms
import os
import json
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw


class OralAutoencoderDataset(torch.utils.data.Dataset):
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
        
        #print image id
        #print("image id", image["id"])
        image_id = image["id"]
        image_name = image["file_name"]
        
        image_path = os.path.join(os.path.dirname(self.images), "oral1", image["file_name"])
        
        image = Image.open(image_path).convert("RGB")
        
        # prova bbox
        x, y, w, h = annotation["bbox"]
        subimage = image.crop((x, y, x+w, y+h))

        if self.transform:
            subimage = self.transform(subimage)

        #image = torch.clamp(image, 0, 1)
        #if image.shape[0] == 1:
        #    image = image.repeat(3, 1, 1)

            
        category = self.categories[annotation["category_id"]]

        return subimage, subimage, category, image_name


if __name__ == "__main__":
    import torchvision

    dataset = OralAutoencoderDataset(
        "data/knn_train.json",
        transform=transforms.Compose([
            transforms.Resize((512, 512), antialias=True),
            transforms.ToTensor()
        ])
    )


    image, image2, label, image_name = dataset.__getitem__(0)
    print(torch.max(image), torch.min(image))
    plt.imshow(image.permute(1, 2, 0))
    plt.show()
    