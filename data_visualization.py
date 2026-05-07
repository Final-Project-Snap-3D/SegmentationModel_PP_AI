import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import VizWiz

# --- setup ---
dataset = VizWiz(
    images_dir='data/train',
    annotations_path='data/annotations/VizWiz_SOD_train_challenge.json'
)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"Dataset size: {len(dataset)} images")

# --- visualize ---
def visualize(dataset, idx):
    image, img_name, mask = dataset[idx]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image)
    axes[0].set_title(f"Image: {img_name}")
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Mask")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f"viz_{idx}.png")   # saves to project root
    plt.close()
    print(f"Saved viz_{idx}.png")


# visualize a few samples
for idx in [0, 1, 2]:
    visualize(dataset, idx)