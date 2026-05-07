import json
import os

from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset

class VizWiz(Dataset):
    def __init__(self, images_dir, annotations_path):
        self.img_names = os.listdir(images_dir)
        self.images_dir = images_dir

        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.img_names)

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

        return image, img_name, mask