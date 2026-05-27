import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Estadístiques d'ImageNet, estàndard per normalitzar imatges RGB
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class DataAugmentation:

    def __init__(self, img_size=256, max_width=None, max_height=None):
        self.img_size = img_size
        # Si no es passen, fem servir img_size per a tots dos (sortida quadrada)
        self.max_width = max_width if max_width is not None else img_size
        self.max_height = max_height if max_height is not None else img_size

    # mantinc aspect ratio-> escalo el costat llarg i omplo amb padding negre fins a (max_h, max_w)... good??
    def _resize_keep_aspect(self):
        return [
            A.LongestMaxSize(max_size=max(self.max_height, self.max_width)),
            A.PadIfNeeded(
                min_height=self.max_height,
                min_width=self.max_width,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
            ),
        ]

    # separo train i val perquè val/test no pot tenir aleatorietat
    def train(self):
        # train: a tope con to
        return A.Compose([
            *self._resize_keep_aspect(),
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
            *self._resize_keep_aspect(),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])