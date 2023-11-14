<div align="center">
  <p>
    <a href="http://snuailab.ai/">
        <img width="75%" src="https://raw.githubusercontent.com/snuailab/assets/main/waffle/icons/waffle_banner.png">
    </a>
  </p>
</div>

Waffle is a framework that lets you use lots of different deep learning tools through just one interface. When it comes to MLOps (machine learning operations), you need to be able to keep up with all the new ideas in deep learning as quickly as possible. But it's hard to do that if you have to write all the code yourself. That's why we started a project to bring together different tools into one framework.

Experience the power of multiple deep learning frameworks at your fingertips with Waffle's seamless integration, unlocking limitless possibilities for your machine learning projects.

# Prerequisites
We've tested Waffle on the following environments:
| OS | Python | PyTorch | Device | Backend | Pass |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Ubuntu 20.04 | 3.9, 3.10 | 1.13.1 | CPU, GPU | All | [![Waffle Hub cpu test](https://github.com/snuailab/waffle_hub/actions/workflows/ci.yaml/badge.svg)](https://github.com/snuailab/waffle_hub/actions/workflows/ci.yaml) |
| Windows | 3.9, 3.10 | 1.13.1 | CPU, GPU | All | [![Waffle Hub cpu test](https://github.com/snuailab/waffle_hub/actions/workflows/ci.yaml/badge.svg)](https://github.com/snuailab/waffle_hub/actions/workflows/ci.yaml) |
| Ubuntu 20.04 | 3.9 | 1.13.1 | Multi GPU | Ultralytics |[![Waffle Hub multi-gpu(ddp) test on self-hosted runner](https://github.com/snuailab/waffle_hub/actions/workflows/ddp.yaml/badge.svg)](https://github.com/snuailab/waffle_hub/actions/workflows/ddp.yaml) |


We recommend using above environments for the best experience.

# Installation
1. Install pytorch and torchvision
    - [PyTorch and TorchVision](https://pytorch.org/get-started/locally/) (We recommend using 1.13.1)
2. Install Waffle Hub
    - `pip install -U waffle-hub`

# Example Usage
We provide both python module and CLI for Waffle Hub.

Following examples do the exact same thing.

## Python Module
```python
from waffle_hub.dataset import Dataset
dataset = Dataset.sample(
  name = "mnist_classification",
  task = "classification",
)
dataset.split(
  train_ratio = 0.8,
  val_ratio = 0.1,
  test_ratio = 0.1
)
export_dir = dataset.export("YOLO")

from waffle_hub.hub import Hub
hub = Hub.new(
  name = "my_classifier",
  task = "classification",
  model_type = "yolov8",
  model_size = "n",
  categories = dataset.get_category_names(),
)
hub.train(
  dataset = dataset,
  epochs = 30,
  batch_size = 64,
  image_size=64,
  device="cpu"
)
hub.inference(
  source=export_dir,
  draw=True,
  device="cpu"
)
```

## CLI
```bash
wd sample --name mnist_classification --task classification
wd split --name mnist_classification --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
wd export --name mnist_classification --data-type YOLO

wh new --name my_classifier --task classification --model-type yolov8 --model-size n --categories [1,2]
wh train --name my_classifier --dataset mnist_classification --epochs 30 --batch-size 64 --image-size 64 --device cpu
wh inference --name my_classifier --source datasets/mnist_classification/exports/YOLO --draw --device cpu
```

# Performance
You can select a model by referring to the performance evaluation below.
### Object Detection

| Backend      |Model and size | image_size | mAPval | Speed (ms) | Parameters (M) | Dataset| GPU|
|--------------|---------------|------------|--------|-----------------|----------------|--------|---|
| ultralytics  | YOLOv8,n      | 640        | 37.3   | 0.99            | 3.2            | COCO    |A100|
| ultralytics  | YOLOv8,s      | 640        | 44.9   | 1.20            | 11.2           | COCO    |A100|
| ultralytics  | YOLOv8,m      | 640        | 50.2   | 1.83            | 25.9           | COCO    |A100|
| ultralytics  | YOLOv8,l      | 640        | 52.9   | 2.39            | 43.7           | COCO    |A100|
| ultralytics  | YOLOv8,x      | 640        | 53.9   | 3.53            | 68.2           | COCO    |A100|
| hugging_face | DETA,base     | 640        | 51.6   | 100             | 52             | COCO 2017|V100|
| hugging_face | DETR,base     | 800        | 42.0   | 35.4            | 41             |COCO 2017|V100|
| hugging_face | DETR,large    | 800        | 43.5   | 50              | 60.7           |COCO 2017|V100|
| hugging_face | Yolos,tiny    | 512 x *    | 28.7   | 1.00            | 6.5            |COCO 2017|1080Ti|
| autocare_dlt | YOLOv5,s      |
| autocare_dlt | YOLOv5,m      |
| autocare_dlt | YOLOv5,l      |


### Classification
| Backend      |Model and size| image_size | Top1 Accuracy | Top5 Accuracy | Speed (ms) | Parameters (M) | Dataset|GPU|
|--------------|-------------|--------------|---------------|---------------|--------------------------|----------------|-------|----|
| ultralytics  | YOLOv8n-cls | 224          | 66.6          | 87.0          | 0.31                     | 2.7            |ImageNet|A100|
| ultralytics  | YOLOv8s-cls | 224          | 72.3          | 91.1          | 0.35                     | 6.4            |ImageNet|A100|
| ultralytics  | YOLOv8m-cls | 224          | 76.4          | 93.2          | 0.62                     | 17.0           |ImageNet|A100|
| ultralytics  | YOLOv8l-cls | 224          | 78.0          | 94.1          | 0.87                     | 37.5           |ImageNet|A100|
| ultralytics  | YOLOv8x-cls | 224          | 78.4          | 94.3          | 1.01                     | 57.4           |ImageNet|A100|
| hugging_face | ViT,tiny    | 224          | 75.45         | 92.844        | 138.538                  | 5.82           |ImageNet|3090|
| hugging_face | ViT,base    | 224          | 84.27         | 96.80         | 846.969                  | 86.6           |ImageNet|3090|
| autocare_dlt | Classifier,s|
| autocare_dlt | Classifier,m|
| autocare_dlt | Classifier,l|

### TextRecognition
| Backend      |Model and size| image_size | mAPval | Speed (ms) | Parameters (M) | Dataset| GPU|
|--------------|--------------|------------|--------|-----------------|----------------|--------|---|
| autocare_dlt | TextRecognition,s|
| autocare_dlt | TextRecognition,m|
| autocare_dlt | TextRecognition,l|
| autocare_dlt | LicencePlateRecognition,s|
| autocare_dlt | LicencePlateRecognition,m|
| autocare_dlt | LicencePlateRecognition,l|

### See our [documentation](https://snuailab.github.io/waffle/) for more information!
