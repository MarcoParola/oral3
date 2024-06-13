import torch
import torchvision.transforms as transforms
import os
import json
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

from src.data1.saliency_classification.dataset import OralClassificationSaliencyDataset


class OralClassificationDataset(torch.utils.data.Dataset):
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

        if self.transform:
            image = self.transform(image)
            
        category = self.categories[annotation["category_id"]]

        return image, category, image_id, image_name


if __name__ == "__main__":
    import torchvision

    '''
    dataset_cropped = OralClassificationDataset(
        "data/train.json", True,
        transform=transforms.Compose([
            transforms.Resize((512, 512), antialias=True),
            transforms.ToTensor()
        ])
    )
    image_cropped, mask = dataset_cropped.__getitem__(5)
    plt.imshow(image_cropped.permute(1, 2, 0))
    plt.savefig("cropped_test.png")


    dataset_not_cropped = OralClassificationDataset(
        "data/train.json", False,
        transform=transforms.Compose([
            transforms.Resize((512, 512), antialias=True),
            transforms.ToTensor()
        ])
    )
    image_not_cropped, mask = dataset_not_cropped.__getitem__(5)
    plt.imshow(image_not_cropped.permute(1, 2, 0))
    plt.savefig("not_cropped_test.png")
    #torchvision.utils.save_image(dataset[1][0], "test.png")
    '''
    dataset = OralClassificationSaliencyDataset(
        "data/train.json", False,
        transform=transforms.Compose([
            transforms.Resize((512, 512), antialias=True),
            transforms.ToTensor()
        ])
    )
    image, label, mask = dataset.__getitem__(0)
    plt.imshow(image.permute(1, 2, 0))
    plt.show()
    plt.imshow(mask.permute(1, 2, 0), cmap='gray', alpha=0.5)
    plt.show()
    print(label)
