import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SegmentationModel, U2Net
from augmentation import DataAugmentation

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        enc_channels = checkpoint.get("enc_channels", (3, 16, 32, 64))
        dec_channels = checkpoint.get("dec_channels", (64, 32, 16))
        model_name   = checkpoint.get("model_name")
        in_channels  = checkpoint.get("in_channels") or 3
        out_channels = checkpoint.get("out_channels") or 1
        state_dict   = checkpoint["model_state_dict"]
    else:
        enc_channels = None
        dec_channels = None
        model_name   = None
        in_channels  = 3
        out_channels = 1
        state_dict   = checkpoint

    if model_name == "U2Net":
        model = U2Net(inChannels=in_channels, outChannels=out_channels).to(device)
    else:
        model = SegmentationModel(
            encChannels=enc_channels or (3, 16, 32, 64),
            decChannels=dec_channels or (64, 32, 16),
            nbClasses=1,
        ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def infer_image(model, image_path, output_dir, transform, threshold, device):
    image = np.array(Image.open(image_path).convert("RGB"))
    original_h, original_w = image.shape[:2]

    tensor = transform(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    mask = (prob > threshold).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask).resize((original_w, original_h), Image.NEAREST)

    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, "mask_" + image_name)
    mask_img.save(output_path)
    return output_path


def run_inference(model_path, input_path, output_dir=None, image_size=512, threshold=0.5, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    if output_dir is None:
        output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(input_path):
        image_paths = sorted([
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ])
        if not image_paths:
            raise ValueError(f"No images found in {input_path}")
    else:
        image_paths = [input_path]

    model = load_model(model_path, device)
    transform = DataAugmentation(img_size=image_size).val_test()

    print(f"Processing {len(image_paths)} image(s) → {output_dir}/")
    for i, image_path in enumerate(image_paths, 1):
        out = infer_image(model, image_path, output_dir, transform, threshold, device)
        print(f"  [{i}/{len(image_paths)}] {os.path.basename(image_path)} → {out}")


def main():
    parser = argparse.ArgumentParser(description="Inferència d'imatges amb el model de segmentació")
    parser.add_argument("--model_path",  required=True, help="ruta al checkpoint del model (.pth)")
    parser.add_argument("--input_path",  required=True, help="imatge o carpeta d'imatges d'entrada")
    parser.add_argument("--output_dir",  default=None,  help="carpeta on guardar les màscares (default: outputs/)")
    parser.add_argument("--image_size",  type=int,   default=512, help="mida d'entrada al model (default: 512)")
    parser.add_argument("--threshold",   type=float, default=0.5, help="llindar de binarització (default: 0.5)")
    parser.add_argument("--device",      type=str,   default=None, help="device: cuda / mps / cpu (auto si no s'especifica)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    run_inference(
        model_path=args.model_path,
        input_path=args.input_path,
        output_dir=args.output_dir,
        image_size=args.image_size,
        threshold=args.threshold,
        device=device,
    )


if __name__ == "__main__":
    main()
