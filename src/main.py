import argparse

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import VizWiz
from model import SegmentationModel
from augmentation import DataAugmentation

# Proposta de main com la sessio1 del lab de MLOps, es pot modificar
def main():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--in_channels", help="canals d'entrada (RGB=3)", type=int, default=3)
    parser.add_argument("--num_classes", help="classes de sortida (binari=1), objecte/no objecte", type=int, default=1)
    parser.add_argument("--base_channels", help="filtres first layer", type=int, default=32)
    # Training
    parser.add_argument("--epochs", help="nombre d'èpoques", type=int, default=10)
    parser.add_argument("--batch_size", help="mida del batch", type=int, default=8)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-3) # FYI: a SAM2 utilitzen reciprocal square-root schedule
    # Data
    parser.add_argument("--image_size", help="mida de les imatges per fer el resize", type=int, default=512) # FYI: el SAM2 es 1024/si cal podriem baixar mes
    parser.add_argument("--train_images_dir", help="ruta imatges train", type=str, default="data/train")
    parser.add_argument("--val_images_dir", help="ruta imatges val", type=str, default="data/val")
    parser.add_argument("--train_annotations", help="ruta JSON train", type=str, default="data/annotations/train.json")
    parser.add_argument("--val_annotations", help="ruta JSON val", type=str, default="data/annotations/val.json")
    
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    aug = DataAugmentation(img_size=args.image_size)
    dataset = VizWiz(args.train_images_dir, args.train_annotations, transform=aug.train())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    val_dataset = VizWiz(args.val_images_dir, args.val_annotations, transform=aug.val())
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = SegmentationModel().to(device)
    criterion = torch.nn.BCEWithLogitsLoss() # Més endavant hauríem de fer BCEWithLogitsLoss amb Dice si les segmentacions no són òptimes en resultats (molt background per exemple)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr) # Adam o Adam W, y palante 

    train_loss_history = []
    val_loss_history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            y_ = model(x)
            
            loss = criterion(y_, y.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        train_loss_history.append(avg_loss)
        
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            n_val_batches = 0
            for x_val, y_val in val_dataloader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                y_val_ = model(x_val)
                loss_val = criterion(y_val_, y_val.float().unsqueeze(1))
                val_loss += loss_val.item()
                n_val_batches += 1

        avg_val_loss = val_loss / n_val_batches
        val_loss_history.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{args.epochs} - Avg Loss: {avg_loss:.4f} - Avg Val Loss: {avg_val_loss:.4f}")

    plt.plot(train_loss_history, label="Training Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
