import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import VizWiz
from augmentation import transform

# --- setup ---
dataset = VizWiz(
    images_dir='data/train',
    annotations_path='data/annotations/VizWiz_SOD_train_challenge.json'
)

dataset_transformed = VizWiz(
    images_dir='data/train', 
    annotations_path='data/annotations/VizWiz_SOD_train_challenge.json', 
    transform=transform)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"Dataset size: {len(dataset)} images")

# --- visualize ---
def visualize(dataset, idx):
    image, mask = dataset[idx]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image.permute(1, 2, 0))
    axes[0].set_title(f"Image: {dataset.get_name(idx)}")
    axes[0].axis('off')

    axes[1].imshow(mask.squeeze(), cmap='gray')
    axes[1].set_title("Mask")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f"viz_{idx}.png")   # saves to project root
    plt.close()
    print(f"Saved viz_{idx}.png")

# --- visualize_transformed ---
def visualize_transformed(dataset, idx):
    image, mask = dataset[idx]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image.permute(1, 2, 0))
    axes[0].set_title(f"Image transformed: {dataset.get_name(idx)}")
    axes[0].axis('off')

    axes[1].imshow(mask.squeeze(), cmap='gray')
    axes[1].set_title("Mask transformed")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f"viz_transformed_{idx}.png")   # saves to project root
    plt.close()
    print(f"Saved viz_transformed_{idx}.png")


# visualize a few samples
for idx in [0, 1, 2]:
    visualize(dataset, idx)

for idx in [0, 1, 2]:
    visualize_transformed(dataset_transformed, idx)