import albumentations as  album
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 512  # mida uniforme de sortida

# Train: augmentació geomètrica + de color
transform =  album.Compose([
     album.Resize(IMG_SIZE, IMG_SIZE),
     album.HorizontalFlip(p=0.5), # mirall horitzontal
     album.RandomRotate90(p=0.3),  # rotació 90° aleatòria
     album.ShiftScaleRotate(shift_limit=0.05, # desplaçament + zoom + rotació lleugera
                       scale_limit=0.1,
                       rotate_limit=15, p=0.4),
     album.RandomBrightnessContrast(p=0.4), # canvi de lluminositat/contrast
     album.GaussianBlur(blur_limit=3, p=0.2), # blur suau (simula desfocament de càmera)
     album.Normalize(mean=(0.485, 0.456, 0.406), # normalització ImageNet
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(), # numpy → tensor (C,H,W) i màscara (H,W)
])

