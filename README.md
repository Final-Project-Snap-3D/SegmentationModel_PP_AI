# SEGMENTATION PIPELINE
## Dataset
Dataset can be downloaded here: https://vizwiz.org/tasks-and-datasets/salient-object-detection/

## Structure
Folder structure should be as follows for the code and dataset for the moment (when project grows separate .py files into according folders (data, src...)

```
vizwiz_salient/
├── data/                # Dataset goes here (not tracked by git)
│   ├── train/           # Training images (.jpg)
│   ├── val/             # Validation images (.jpg)
│   ├── test/            # Test images (.jpg)
│   └── annotations/     # JSON annotation files
│       ├── train.json
|       ├── test.json
│       └── val.json
├── dataset.py
├── requirements.txt
└── data_visualization.py
```
