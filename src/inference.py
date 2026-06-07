import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SegmentationModel, U2Net
from augmentation import DataAugmentation


def run_inference(model_path, image_path, output_path, image_size=512, threshold=0.5, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

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

    image = np.array(Image.open(image_path).convert("RGB"))
    original_h, original_w = image.shape[:2]

    transform = DataAugmentation(img_size=image_size).val_test()
    tensor = transform(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    mask = (prob > threshold).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask).resize((original_w, original_h), Image.NEAREST)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    mask_img.save(output_path)
    print(f"Mask saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Inferència d'una sola imatge")
    parser.add_argument("--model_path",  required=True, help="ruta al checkpoint del model (.pth)")
    parser.add_argument("--image_path",  required=True, help="ruta a la imatge d'entrada")
    parser.add_argument("--output_path", required=True, help="ruta on guardar la màscara predita (.png)")
    parser.add_argument("--image_size",  type=int,   default=512,  help="mida d'entrada al model (default: 512)")
    parser.add_argument("--threshold",   type=float, default=0.5,  help="llindar de binarització (default: 0.5)")
    parser.add_argument("--device",      type=str,   default=None, help="device: cuda / mps / cpu (auto si no s'especifica)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    run_inference(
        model_path=args.model_path,
        image_path=args.image_path,
        output_path=args.output_path,
        image_size=args.image_size,
        threshold=args.threshold,
        device=device,
    )


if __name__ == "__main__":
    main()
