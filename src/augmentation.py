# import albumentations as  album
# from albumentations.pytorch import ToTensorV2
#
# IMG_SIZE = 512  # mida uniforme de sortida
#
# # Train: augmentació geomètrica + de color
# transform =  album.Compose([
#      album.Resize(IMG_SIZE, IMG_SIZE),
#      album.HorizontalFlip(p=0.5), # mirall horitzontal
#      album.RandomRotate90(p=0.3),  # rotació 90° aleatòria
#      album.ShiftScaleRotate(shift_limit=0.05, # desplaçament + zoom + rotació lleugera
#                        scale_limit=0.1,
#                        rotate_limit=15, p=0.4),
#      album.RandomBrightnessContrast(p=0.4), # canvi de lluminositat/contrast
#      album.GaussianBlur(blur_limit=3, p=0.2), # blur suau (simula desfocament de càmera)
#      album.Normalize(mean=(0.485, 0.456, 0.406), # normalització ImageNet
#                 std=(0.229, 0.224, 0.225)),
#     ToTensorV2(), # numpy → tensor (C,H,W) i màscara (H,W)
# ])
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Estadístiques d'ImageNet, estàndard per normalitzar imatges RGB
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class DataAugmentation:
    
    def __init__(self, img_size=256):
        self.img_size = img_size

    # separo train i val perquè val/test no pot tenir aleatorietat 
    def train(self):
        # train: a tope con to
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.1), 
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),  
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    def val_test(self):
        # val/test: només resize + normalize per reproduir les mètriques entre epochs
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])