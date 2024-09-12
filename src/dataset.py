import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ProductImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, os.path.basename(self.data.iloc[idx]['image_link']))
        image = Image.open(img_name).convert('RGB')
        
        image = self.transform(image)
        
        entity_name = self.data.iloc[idx]['entity_name']
        entity_value = self.data.iloc[idx]['entity_value'] if 'entity_value' in self.data.columns else ""
        
        return image, entity_name, entity_value

def get_dataloader(csv_file, img_dir, batch_size=32, shuffle=True):
    dataset = ProductImageDataset(csv_file, img_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)