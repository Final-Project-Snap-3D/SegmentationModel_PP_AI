import json
import os

from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class VizWiz(Dataset):
    # He afegit el paràmetre transform per poder convertir imatge i màscara a tensors des de fora
    def __init__(self, images_dir, annotations_path, transform=None):
        self.img_names = os.listdir(images_dir)
        self.images_dir = images_dir
        self.transform = transform 

        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.img_names)

    # He tret img_name del retorn per no carregar el DataLoader, i ho fem amb get_name(idx)
    def get_name(self, idx):
        return self.img_names[idx]

    def __getitem__(self, idx):
        #Get image
        img_name = self.img_names[idx]
        img_path = os.path.join(self.images_dir, img_name)

        image = Image.open(img_path).convert('RGB')

        #Get mask
        annotation = self.annotations[img_name]
        height, width = annotation['Ground Truth Dimensions']

        mask = np.zeros((height, width), dtype=np.uint8)
        salient_object = annotation['Salient Object']
        numpy_list = [np.array(polygon) for polygon in salient_object]
        mask = cv2.fillPoly(mask, numpy_list, color=1)

        # He modificat la sortida perquè retorni tensors enlloc de PIL/numpy, pel DataLoader
        if self.transform is not None:
            # rep numpy i retorna tensors
            transformed = self.transform(image=np.array(image), mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Imatge a float [0,1] (C,H,W) i màscara a long (H,W)
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask