import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import VizWiz
from model import SegmentationModel, U2Net
from augmentation import DataAugmentation
from wandb_logger import WandbLogger
from utils import TaskType
from losses import BCEDiceLoss

from test_evaluation import run_evaluation

def compute_iou_dice_metrics(y_pred, y_true, eps=1e-7):
    """
    Compute IoU and Dice metrics for segmentation.
    
    Args:
        y_pred: Predicted logits from model
        y_true: Ground truth binary masks
        eps: Small epsilon value for numerical stability
    
    Returns:
        iou: IoU scores per sample
        dice: Dice scores per sample
    """
    y_prob = torch.sigmoid(y_pred)
    y_hat = (y_prob > 0.5).float()
    
    intersection = (y_hat * y_true).sum(dim=(1, 2, 3))
    union = y_hat.sum(dim=(1, 2, 3)) + y_true.sum(dim=(1, 2, 3)) - intersection
    
    iou = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (y_hat.sum(dim=(1, 2, 3)) + y_true.sum(dim=(1, 2, 3)) + eps)
    
    return iou, dice, y_hat

# Proposta de main com la sessio1 del lab de MLOps, es pot modificar
def main():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--model_name", help="nom del model a entrenar", type=str, default="U2") # Per defecte U2, "U" per utilitzar U-Net
    parser.add_argument("--in_channels", help="canals d'entrada (RGB=3)", type=int, default=3)
    parser.add_argument("--num_classes", help="classes de sortida (binari=1), objecte/no objecte", type=int, default=1)
    parser.add_argument("--base_channels", help="filtres first layer", type=int, default=32)
    # Training
    parser.add_argument("--epochs", help="nombre d'èpoques", type=int, default=100)
    parser.add_argument("--batch_size", help="mida del batch", type=int, default=32)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-3) # FYI: a SAM2 utilitzen reciprocal square-root schedule
    parser.add_argument("--log_image_every", help="log validation images every N epochs", type=int, default=5)
    # Data
    parser.add_argument("--image_size", help="mida de les imatges per fer el resize", type=int, default=512) # FYI: el SAM2 es 1024/si cal podriem baixar mes
    parser.add_argument("--train_images_dir", help="ruta imatges train", type=str, default="data/train")
    parser.add_argument("--val_images_dir", help="ruta imatges val", type=str, default="data/val")
    parser.add_argument("--train_annotations", help="ruta JSON train", type=str, default="data/annotations/VizWiz_SOD_train_challenge.json")
    parser.add_argument("--val_annotations", help="ruta JSON val", type=str, default="data/annotations/VizWiz_SOD_val_challenge.json")
    # Test
    parser.add_argument("--test_images_dir", help="ruta imatges test", type=str, default="data/test")
    parser.add_argument("--test_annotations", help="ruta JSON test", type=str, default="data/annotations/VizWiz_SOD_test_challenge.json")
    # Checkpoints
    parser.add_argument("--checkpoint_dir", help="ruta per guardar checkpoints", type=str, default="checkpoints")
    parser.add_argument("--resume", help="ruta a un checkpoint (last.pt) per reprendre el training", type=str, default=None)

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
    
    val_dataset = VizWiz(args.val_images_dir, args.val_annotations, transform=aug.val_test())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Model, loss, optimizer
    b = args.base_channels
    if args.model_name == "U2":
        model = U2Net(
            inChannels=args.in_channels,
            outChannels=args.num_classes,
        ).to(device)
    else:
        model = SegmentationModel(
            encChannels=(args.in_channels, b, b * 2, b * 4, b * 8),
            decChannels=(b * 8, b * 4, b * 2, b),
            nbClasses=args.num_classes,
        ).to(device)
    
    criterion = BCEDiceLoss(bce_weight=0.3, dice_weight=0.7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr) # Adam o Adam W, y palante 

    # Logger
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger = WandbLogger(task=TaskType.SEGMENTATION, model=model)

    # Training loop
    print("\nStarting training...\n")
    best_val_loss = float('inf')
    best_model_path = checkpoint_dir / "best_model.pt"
    last_model_path = checkpoint_dir / "last.pt"
    eps = 1e-7
    start_epoch = 0

    # Resume from checkpoint if requested
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        else:
            print("  ⚠ checkpoint has no optimizer state — AdamW will restart cold")
        start_epoch = int(ckpt.get('epoch', 0))
        best_val_loss = float(ckpt.get('best_val_loss', float('inf')))
        print(f"  ↳ resuming at epoch {start_epoch+1}/{args.epochs}, best_val_loss={best_val_loss:.4f}")

    for epoch in range(start_epoch, args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for i, (x, y) in enumerate(train_loader, 1):
            if i % 500 == 0:
                print(f"  Batch {i}/{len(train_loader)}")
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            y_true = y.float().unsqueeze(1)
            if isinstance(y_pred, (list, tuple)):
                loss = sum(criterion(pred, y_true) for pred in y_pred) / len(y_pred)
            else:
                loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_iou_sum = 0.0
        val_dice_sum = 0.0
        val_samples = 0
        sample_images = None
        sample_masks = None
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                y_true = y.float().unsqueeze(1)
                if isinstance(y_pred, (list, tuple)):
                    loss = sum(criterion(pred, y_true) for pred in y_pred) / len(y_pred)
                    y_pred_for_metrics = y_pred[0]
                else:
                    loss = criterion(y_pred, y_true)
                    y_pred_for_metrics = y_pred
                val_loss += loss.item()

                iou, dice, y_hat = compute_iou_dice_metrics(y_pred_for_metrics, y_true, eps)

                val_iou_sum += iou.sum().item()
                val_dice_sum += dice.sum().item()
                val_samples += x.size(0)

                if sample_images is None:
                    sample_images = x[:3].cpu()
                    sample_masks = y_hat[:3].cpu()

        val_loss /= len(val_loader)
        val_iou = val_iou_sum / val_samples
        val_dice = val_dice_sum / val_samples
        
        # Log
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f}"
        )
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'val_dice': val_dice,
        }
        if (epoch + 1) % args.log_image_every == 0 and sample_images is not None and sample_masks is not None:
            metrics.update(logger.build_segmentation_images(
                images=sample_images,
                masks=sample_masks,
                epoch=epoch + 1,
                max_items=3,
            ))
        logger.log_metrics(metrics, step=epoch+1)
        if 'val/predictions' in metrics:
            print("  ✓ Logged 3 validation segmentation samples")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = checkpoint_dir / "best_model.pt"
            logger.save_checkpoint(str(best_model_path))
            print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")

        # Always save last.pt (resumable: model + optimizer + epoch + best_val_loss)
        logger.save_checkpoint(
            str(last_model_path),
            optimizer=optimizer,
            epoch=epoch + 1,
            best_val_loss=best_val_loss,
        )
    
    # Save final model
    final_model_path = checkpoint_dir / "final_model.pt"
    logger.save_checkpoint(str(final_model_path))
    print(f"\n✓ Training completed!")
    print(f"  Checkpoints saved to: {checkpoint_dir}")
    
    logger.log_model(str(best_model_path), name="best_model")
    logger.log_model(str(final_model_path), name="final_model")
    logger.finish()

    # Test evaluation with best model
    if best_model_path.exists() and Path(args.test_images_dir).exists():
        print(f"\nRunning test evaluation with best model: {best_model_path}")
        test_metrics = run_evaluation(
            model_path=str(best_model_path),
            images_dir=args.test_images_dir,
            annotations=args.test_annotations,
            image_size=args.image_size,
            batch_size=args.batch_size,
        )
        if test_metrics is not None:
            print("\n=== Test Evaluation Results ===")
            for k, v in test_metrics.items():
                print(f"  {k.upper():12s}: {v:.4f}")
            print("================================\n")

if __name__ == "__main__":
    main()
