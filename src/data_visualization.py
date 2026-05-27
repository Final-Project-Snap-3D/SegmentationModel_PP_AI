import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import VizWiz
from augmentation import DataAugmentation, IMAGENET_MEAN, IMAGENET_STD

# --- setup ---
dataset = VizWiz(
    images_dir='data/train',
    annotations_path='data/annotations/VizWiz_SOD_train_challenge.json'
)

# amb augmentation
aug = DataAugmentation(img_size=512)
dataset_transformed = VizWiz(
    images_dir='data/train',
    annotations_path='data/annotations/VizWiz_SOD_train_challenge.json',
    transform=aug.train()
)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"Dataset size: {len(dataset)} images")

def denormalize(image):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1)


# overlay de màscara en vermell sobre la imatge
def make_overlay(image_np, mask_np, color=(1.0, 0.0, 0.0), alpha=0.5):
    # Si raw dataset, image (mida nativa) i mask (mida anotació) poden no coincidir → redimensionem la màscara
    if mask_np.shape[:2] != image_np.shape[:2]:
        h, w = image_np.shape[:2]
        mask_np = cv2.resize(mask_np.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    color_layer = np.zeros_like(image_np)
    color_layer[mask_np > 0] = color
    blended = (1 - alpha) * image_np + alpha * color_layer
    return np.where(mask_np[..., None] > 0, blended, image_np).clip(0, 1)


# --- visualize ---
def visualize(dataset, idx, prefix="viz", do_denormalize=False, show_mask_overlay=False):
    image, mask = dataset[idx]

    # Converteix tensor (C,H,W) a numpy (H,W,C) per a matplotlib
    if isinstance(image, torch.Tensor):
        if do_denormalize:
            image = denormalize(image)
        image = image.permute(1, 2, 0).cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    n_panels = 3 if show_mask_overlay else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    axes[0].imshow(image)
    axes[0].set_title(f"Image: {dataset.get_name(idx)}")
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Mask")
    axes[1].axis('off')

    if show_mask_overlay:
        axes[2].imshow(make_overlay(image, mask))
        axes[2].set_title("Overlay")
        axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f"{prefix}_{idx}.png")   # saves to project root
    plt.close()
    print(f"Saved {prefix}_{idx}.png")


# visualize a few samples (raw)
for idx in [0, 1, 2]:
    visualize(dataset, idx, show_mask_overlay=True)

# visualize amb augmentació aplicada (overlay per comprovar alineació)
for idx in [0, 1, 2]:
    visualize(dataset_transformed, idx, prefix="viz_transformed",
              do_denormalize=True, show_mask_overlay=True)
