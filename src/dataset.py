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
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)

        # Només les imatges que tenen anotació i que completen el format requerit
        valid_images = []
        for n in os.listdir(images_dir):
            if n in self.annotations:
                # Check that annotation has required keys
                ann = self.annotations[n]
                if 'Salient Object' in ann and 'Ground Truth Dimensions' in ann:
                    valid_images.append(n)
        self.img_names = valid_images
        
        self.images_dir = images_dir
        self.annotations_path = annotations_path
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    # He tret img_name del retorn per no carregar el DataLoader, i ho fem amb get_name(idx)
    def get_name(self, idx):
        return self.img_names[idx]

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.images_dir, img_name)

        image = Image.open(img_path).convert('RGB')

        annotation = self.annotations[img_name]
        height, width = annotation['Ground Truth Dimensions']

        # resize IMAGE to match annotation
        image = image.resize((width, height), Image.BILINEAR)

        mask = np.zeros((height, width), dtype=np.uint8)
        salient_object = annotation['Salient Object']
        numpy_list = [np.array(polygon) for polygon in salient_object]
        mask = cv2.fillPoly(mask, numpy_list, color=1)

        if self.transform is not None:
            transformed = self.transform(image=np.array(image), mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask