import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


# --- visualize ---
def visualize(dataset, idx, prefix="viz", do_denormalize=False):
    image, mask = dataset[idx]

    # Converteix tensor (C,H,W) a numpy (H,W,C) per a matplotlib
    if isinstance(image, torch.Tensor):
        if do_denormalize:
            image = denormalize(image)
        image = image.permute(1, 2, 0).cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image)
    axes[0].set_title(f"Image: {dataset.get_name(idx)}")
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Mask")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f"{prefix}_{idx}.png")   # saves to project root
    plt.close()
    print(f"Saved {prefix}_{idx}.png")


# visualize a few samples (raw)
for idx in [0, 1, 2]:
    visualize(dataset, idx)

# visualize amb augmentació aplicada
for idx in [0, 1, 2]:
    visualize(dataset_transformed, idx, prefix="viz_transformed", do_denormalize=True)
