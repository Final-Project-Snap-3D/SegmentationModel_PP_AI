import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import VizWiz
from model import SegmentationModel
from augmentation import DataAugmentation
from wandb_logger import WandbLogger
from utils import TaskType


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
    parser.add_argument("--train_annotations", help="ruta JSON train", type=str, default="data/annotations/VizWiz_SOD_train_challenge.json")
    parser.add_argument("--val_annotations", help="ruta JSON val", type=str, default="data/annotations/VizWiz_SOD_val_challenge.json")
    # Checkpoints
    parser.add_argument("--checkpoint_dir", help="ruta per guardar checkpoints", type=str, default="checkpoints")

    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Data loading
    aug = DataAugmentation(img_size=args.image_size)
    train_dataset = VizWiz(args.train_images_dir, args.train_annotations, transform=aug.train())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    val_dataset = VizWiz(args.val_images_dir, args.val_annotations, transform=aug.val())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Model, loss, optimizer
    model = SegmentationModel().to(device)
    criterion = torch.nn.BCEWithLogitsLoss() # Més endavant hauríem de fer BCEWithLogitsLoss amb Dice si les segmentacions no són òptimes en resultats (molt background per exemple)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr) # Adam o Adam W, y palante 

    # Logger
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger = WandbLogger(task=TaskType.SEGMENTATION, model=model)

    # Training loop
    print("\nStarting training...\n")
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for i, (x, y) in enumerate(train_loader, 1):
            if i % 10 == 0:
                print(f"  Batch {i}/{len(train_loader)}")
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y.float().unsqueeze(1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Log
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        logger.log_metrics({
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, step=epoch+1)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = checkpoint_dir / "best_model.pt"
            logger.save_checkpoint(str(best_model_path))
            print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
    
    # Save final model
    final_model_path = checkpoint_dir / "final_model.pt"
    logger.save_checkpoint(str(final_model_path))
    print(f"\n✓ Training completed!")
    print(f"  Checkpoints saved to: {checkpoint_dir}")
    
    logger.finish()


if __name__ == "__main__":
    main()
