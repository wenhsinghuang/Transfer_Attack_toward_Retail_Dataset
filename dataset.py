import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import cv2

# The Freiburg Groceries Dataset
annotations_files = ['splits/train0.txt','splits/train0.txt','splits/train0.txt','splits/train0.txt','splits/train0.txt']
img_dir = 'images/'

class GroceryDataset(Dataset):
    def __init__(self, annotations_files, img_dir, transform=None, target_transform=None):
        
        self.img_paths = []
        self.img_labels = []
        for annotation_file in annotations_files:
            with open(annotation_file) as f:
                for line in f.readlines():
                    self.img_paths.append(line.split()[0])
                    self.img_labels.append(int(line.split()[1]))
                    
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.img_dir + self.img_paths[idx]), cv2.COLOR_BGR2RGB)
        image = np.array(image)
        label = np.array(self.img_labels[idx])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


transform = transforms.Compose([
    transforms.ToTensor() # !!!
])

train_grocery_dataset = GroceryDataset(annotations_files, img_dir, transform)
image, label = train_grocery_dataset[0]

# Use DATA loader to load data ...


# SKU-110K