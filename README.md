# SEGMENTATION PIPELINE

Pipeline de segmentació binària (salient object detection) sobre el dataset VizWiz.

## Dataset

Descarrega: https://vizwiz.org/tasks-and-datasets/salient-object-detection/

### Format de les dades

- **Imatges**: `.jpg` RGB, mides variables.
- **Anotacions**: un JSON per split. Cada entrada té forma:

```json
"VizWiz_train_00000000.jpg": {
    "Full Screen": false,
    "Total Polygons": 1,
    "Ground Truth Dimensions": [H, W],
    "Salient Object": [[[x1, y1], [x2, y2], ...]]
}
```

- `Ground Truth Dimensions`: mida `[alçada, amplada]` a la que es refereixen els polígons.
- `Salient Object`: llista de polígons (vèrtexs `[x, y]`) que delimiten l'objecte. Es converteixen a màscara binària amb `cv2.fillPoly`.
- El `VizWiz` dataset filtra automàticament les imatges sense entrada al JSON.

## Estructura de carpetes

```
vizwiz_salient/
├── data/                       # no tracked per git
│   ├── train/                  # imatges .jpg
│   ├── val/
│   ├── test/
│   └── annotations/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── src/
│   ├── dataset.py
│   ├── augmentation.py
│   ├── model.py
│   ├── main.py
│   └── data_visualization.py
├── requirements.txt
└── README.md
```

## Data augmentation

Implementat a `augmentation.py` amb la classe `DataAugmentation`, que exposa dos pipelines:

- **`train()`**: transformacions geomètriques (`HorizontalFlip`, `RandomRotate90`, `ShiftScaleRotate`) + visuals (`RandomBrightnessContrast`, `HueSaturationValue`, `GaussianBlur`, `GaussNoise`) + `Normalize` (ImageNet) + `ToTensorV2`.
- **`val()`**: només `Resize` + `Normalize` + `ToTensorV2`. Sense aleatorietat per garantir mètriques comparables entre epochs.

Ús:

```python
from augmentation import DataAugmentation
aug = DataAugmentation(img_size=512)
train_ds = VizWiz(..., transform=aug.train())
val_ds   = VizWiz(..., transform=aug.val())
```

## Model (a emplenar)

`model.py` defineix una arquitectura UNet en 3 classes:

- **`Encoder (Downsampling)`**: ...
- **`Decoder (Upsampling)`**: ...
- **`SegmentationModel`**: combina encoder i decoder. Sortida: logits per píxel.

## Entrenament

Execució bàsica:

```bash
python src/main.py
```

Amb arguments personalitzats:

```bash
python src/main.py --image_size 512 --batch_size 16 --epochs 20 --lr 1e-3
```

### Arguments

- `--in_channels` (default `3`) — canals d'entrada (RGB).
- `--num_classes` (default `1`) — classes de sortida (binari).
- `--base_channels` (default `32`) — filtres de la primera capa.

- `--epochs` (default `10`) — nombre d'èpoques.
- `--batch_size` (default `8`) — mida del batch.
- `--lr` (default `1e-3`) — learning rate.

- `--image_size` (default `512`) — mida a la qual es redimensionen les imatges.
- `--train_images_dir`, `--val_images_dir` — rutes a les carpetes d'imatges.
- `--train_annotations`, `--val_annotations` — rutes als JSON d'anotacions.

## Visualització

Per inspeccionar mostres del dataset:

```bash
python src/data_visualization.py
```

Genera `viz_{i}.png` (originals) i `viz_transformed_{i}.png` (amb augmentació, desnormalitzades).

## Instal·lació

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
