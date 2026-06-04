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

## Model 

Implementat a `model.py`amb la classe SegmentationModel, seguint la guia del lab de UNet. Adaptat algunes coses (padding, eliminar center cropping...). Batch Normalization afegida també entre capa convolucional i ReLU ja que recomanat quan s'entrena desde zero.

## Entrenament

Afegits els loops d'entrenament i validació, generant les gràfiques de loss per ambdues.

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

## Testing training loop with only n test samples and n validation samples

Create folders inside data named one_image_val / one_image_train and place there the samples you want to use

```bash
python src/main.py --batch_size 1 --train_images_dir "data/one_image_train" --val_images_dir "data/one_image_val"
```

## YOLO26 (experimental)

Pipeline alternatiu utilitzant YOLO26 d'Ultralytics per instance segmentation. Tractem la salient object detection com a segmentació d'una sola classe.

### Instal·lació

```bash
pip install ultralytics
```

Si PyTorch no detecta la GPU (`torch.cuda.is_available()` retorna `False`), reinstal·la-ho amb suport CUDA. Per Python 3.13:

```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Conversió de dades

YOLO no llegeix els JSON de VizWiz. Cal convertir-los a format YOLO (un `.txt` per imatge amb polígons normalitzats). Una sola execució:

```bash
python src/convert_vizwiz_to_yolo.py
```

Això genera `data_yolo/` com a directori germà de `data/`:

```
data_yolo/
├── images/
│   ├── train/      # còpies de .jpg
│   └── val/
├── labels/
│   ├── train/      # .txt amb polígons normalitzats [0,1]
│   └── val/
└── vizwiz.yaml     # config del dataset
```

### Entrenament

```bash
python src/train_yolo.py
```

Defaults: `yolo26s-seg`, `batch=4`, `imgsz=512`, `epochs=100`, AMP activat. Pensat per GTX 1060 6GB.

### Arguments

- `--model` (default `yolo26s-seg.pt`) — `yolo26n-seg.pt` (mínim) | `yolo26s-seg.pt` | `yolo26m-seg.pt`.
- `--epochs` (default `100`).
- `--batch` (default `4`) — pujar a 8 si la VRAM ho permet, baixar a 2 si OOM.
- `--imgsz` (default `512`).
- `--device` (default `0`) — índex de la GPU CUDA a utilitzar (`0` = primera GPU). Passar `cpu` per entrenar en CPU.
- `--project`, `--name` — ruta de sortida (`runs/yolo_vizwiz/exp/`).

### Mètriques

YOLO loga `metrics/mAP50(M)` i `metrics/mAP50-95(M)` (mAP sobre màscares) a `runs/yolo_vizwiz/exp/results.csv`. Per comparar amb el U-Net via Dice/IoU cal un script de post-evaluació que uneixi totes les instàncies predites en una sola màscara binària per imatge.