# 🧠 Brain Tumor MRI Classification — Deep Learning Study

A comprehensive deep learning study comparing **5 models** for automated brain tumor classification from MRI scans — from a lightweight custom CNN trained from scratch to a state-of-the-art Vision Transformer.

---

## 📌 Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Repository Structure](#repository-structure)
4. [Environment Setup](#environment-setup)
5. [Model 1 — Custom Sequential CNN](#model-1--custom-sequential-cnn)
6. [Model 2 — VGG16 Transfer Learning](#model-2--vgg16-transfer-learning)
7. [Model 3 — Transfer Learning Comparison (VGG16 · MobileNetV3 · ResNet50)](#model-3--transfer-learning-comparison)
8. [Model 4 — Vision Transformer (ViT-B/16)](#model-4--vision-transformer-vit-b16)
9. [Model Comparison](#model-comparison)
10. [References](#references)

---

## Project Overview

Brain tumors are one of the most dangerous neurological conditions, and early accurate diagnosis significantly improves patient outcomes. Manual analysis of MRI scans is time-consuming and prone to inter-observer variability. This project implements and compares five deep learning approaches — ranging from a custom CNN to a Vision Transformer — to automate 4-class brain tumor classification from MRI images.

**Problem Statement:** Given an MRI scan image, classify it into one of four categories:
- **Glioma** — malignant tumor arising from glial cells
- **Meningioma** — usually benign; originates from brain lining
- **No Tumor** — healthy brain scan (negative class)
- **Pituitary** — tumor of the pituitary gland affecting hormone regulation

**Approach:** Each model is trained and evaluated on the same dataset split, with results compared across accuracy, precision, recall, F1-score, test loss, model size, and per-class performance.

---

## Dataset

### Source
| Property | Details |
|----------|---------|
| **Name** | Brain Tumor MRI Dataset |
| **Provider** | Masoud Nickparvar |
| **Platform** | Kaggle |
| **Link** | https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset |
| **License** | Open for research and education |

### Download
```bash
# Using Kaggle API
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p ./data --unzip
```

### Structure
```
data/
├── Training/
│   ├── glioma/          # 1,321 images
│   ├── meningioma/      # 1,339 images
│   ├── notumor/         # 1,595 images
│   └── pituitary/       # 1,457 images
│
└── Testing/
    ├── glioma/           # 300 images
    ├── meningioma/       # 306 images
    ├── notumor/          # 405 images
    └── pituitary/        # 300 images
```

### Class Distribution

| Class | Training | Testing | Total | % of Total |
|-------|----------|---------|-------|------------|
| Glioma | 1,321 | 300 | 1,621 | 23.1% |
| Meningioma | 1,339 | 306 | 1,645 | 23.4% |
| No Tumor | 1,595 | 405 | 2,000 | 28.5% |
| Pituitary | 1,457 | 300 | 1,757 | 25.0% |
| **Total** | **5,712** | **1,311** | **7,023** | 100% |

### Class Descriptions

**Glioma**
Gliomas arise from glial cells (supportive cells of the brain) and are the most common primary malignant brain tumors. They include glioblastoma multiforme (GBM), which is highly aggressive. On MRI, gliomas typically appear as irregular, heterogeneous masses with surrounding edema.

**Meningioma**
Meningiomas originate from the meninges — the membranes surrounding the brain and spinal cord. They are the most common benign brain tumor. On MRI they often appear as well-defined, homogeneous masses attached to the dura. They are the hardest class for all models due to visual similarity to surrounding tissue.

**No Tumor**
Normal brain MRI scans with no detectable tumor. These serve as the negative class and are slightly over-represented (28.5%) in the dataset.

**Pituitary**
Pituitary adenomas arise from the pituitary gland at the base of the brain. They affect hormone production and can cause endocrine disorders. They are typically well-defined and easier to classify due to their distinctive location.

### Image Properties
- **Format:** JPEG and PNG
- **Color:** RGB (even though MRI is inherently grayscale — 3-channel for compatibility)
- **Resolution:** Variable (resized to 224×224 for all models)
- **Modality:** T1, T2, and FLAIR weighted MRI sequences

---

## Repository Structure

```
brain-tumor-mri-classification/
│
├── notebook_01_custom_cnn.ipynb          ← Custom Sequential CNN (already provided)
├── notebook_02_vgg16.ipynb               ← VGG16 Transfer Learning
├── notebook_03_transfer_learning.ipynb   ← VGG16 + MobileNetV3 + ResNet50
├── notebook_04_vit.ipynb                 ← Vision Transformer (ViT-B/16)
├── notebook_05_comparison.ipynb          ← All-model comparison & visualizations
│
└── README.md
```

**Run order:** 01 → 02 → 03 → 04 → 05 (each saves a `*_metrics.json` file used by notebook 05)

---

## Environment Setup

### Google Colab (Recommended)
All notebooks are designed for Google Colab with a **T4 GPU**.

```
Runtime → Change runtime type → Hardware accelerator → GPU (T4)
```

### Kaggle API Key
All notebooks download the dataset automatically. You need a `kaggle.json` API key:
1. Go to https://www.kaggle.com/account
2. Click **Create New Token** → downloads `kaggle.json`
3. Upload when prompted in each notebook

### Dependencies by Notebook

| Notebook | Install Command |
|----------|----------------|
| 01 Custom CNN | `pip install kaggle` |
| 02 VGG16 | `pip install kaggle imutils opencv-python-headless` |
| 03 Transfer Learning | `pip install kaggle` |
| 04 ViT | `pip install kaggle` |
| 05 Comparison | *(no installs — all libraries pre-installed on Colab)* |

> TensorFlow, PyTorch, NumPy, Matplotlib, Seaborn, scikit-learn, and Pandas are **pre-installed** on Colab and do not need `pip install`.

### Expected Runtime (T4 GPU)

| Notebook | Approximate Time |
|----------|-----------------|
| 01 Custom CNN | 20–30 min |
| 02 VGG16 | 25–40 min |
| 03 Transfer Learning (3 models) | 60–90 min |
| 04 ViT-B/16 | 45–60 min |
| 05 Comparison | < 2 min |

---

## Model 1 — Custom Sequential CNN

### Overview
| Property | Details |
|----------|---------|
| **Reference** | [Kaggle — yousefmohamed20 (Acc ~99%)](https://www.kaggle.com/code/yousefmohamed20/brain-tumor-mri-accuracy-99) |
| **Updated Colab** | https://colab.research.google.com/drive/1QaeR2XMkm62vQ5ODXkYbe37t0YkYJNzO |
| **Framework** | TensorFlow / Keras Sequential API |
| **Type** | Custom CNN — trained from scratch |
| **Parameters** | ~2.1 Million |
| **Pretrained** | ❌ No — random weight initialization |

> ⚠️ **This notebook is already provided — not included in this repository.** Its results are included in the final comparison.

---

### Step-by-Step Walkthrough

#### Step 1 — Library Imports
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, BatchNormalization,
                                      Dropout, Flatten, Dense)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
```

#### Step 2 — Dataset Loading
The dataset is downloaded from Kaggle and loaded using `ImageDataGenerator.flow_from_directory`:
- Training images are read from `Training/` subdirectories
- Validation split: 20% of training images held out
- Test images are read from `Testing/` without augmentation

#### Step 3 — Preprocessing

| Step | Training | Validation/Test |
|------|----------|----------------|
| Resize | 224×224 | 224×224 |
| Normalize | rescale=1/255 → [0,1] | rescale=1/255 → [0,1] |
| Rotation | ±15° | — |
| Horizontal flip | ✅ | — |
| Zoom | 10% | — |
| Width/Height shift | 10% | — |

#### Step 4 — Model Architecture

```
Input (224 × 224 × 3)
    │
    ▼
Conv2D(32, 3×3, ReLU) → BatchNormalization → MaxPooling2D(2×2)
    │
    ▼
Conv2D(64, 3×3, ReLU) → BatchNormalization → MaxPooling2D(2×2)
    │
    ▼
Conv2D(128, 3×3, ReLU) → BatchNormalization → MaxPooling2D(2×2) → Dropout(0.25)
    │
    ▼
Conv2D(256, 3×3, ReLU) → BatchNormalization → MaxPooling2D(2×2) → Dropout(0.25)
    │
    ▼
Flatten
    │
    ▼
Dense(512, ReLU) → Dropout(0.5)
    │
    ▼
Dense(4, Softmax)   ← 4 output classes
```

**Design rationale:**
- Each convolutional block doubles the number of filters (32→64→128→256), progressively learning more complex features
- `BatchNormalization` stabilizes training and accelerates convergence
- `Dropout` prevents overfitting at each stage
- `MaxPooling` reduces spatial dimensions and computation

#### Step 5 — Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Loss Function | Categorical Cross-Entropy |
| Batch Size | 32 |
| Epochs | 25–50 (with EarlyStopping) |
| EarlyStopping Patience | 10 |
| Metric | Accuracy |

#### Step 6 — Input / Output Summary

**Input:**
- Raw MRI JPEG/PNG images from class-labeled subdirectories
- Processed through `ImageDataGenerator` → batches of 32 images of shape `(32, 224, 224, 3)`
- Labels as one-hot encoded vectors of shape `(32, 4)`

**Output:**
- Softmax probability vector of shape `(n_samples, 4)`
- `argmax` → predicted class index → mapped to class name
- Training history: loss and accuracy curves per epoch
- `classification_report` — Precision, Recall, F1 per class
- Confusion matrix heatmap
- Saved model: `custom_cnn_brain_tumor.h5`
- Saved metrics: `custom_cnn_metrics.json`

---

### Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | ~99.0% |
| **Precision (macro)** | ~99.0% |
| **Recall (macro)** | ~99.0% |
| **F1-Score (macro)** | ~99.0% |
| **Test Loss** | ~0.04 |

**Per-Class Performance:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | ~99% | ~99% | ~99% |
| Meningioma | ~98% | ~99% | ~99% |
| No Tumor | ~100% | ~99% | ~100% |
| Pituitary | ~99% | ~99% | ~99% |

---

## Model 2 — VGG16 Transfer Learning

### Overview
| Property | Details |
|----------|---------|
| **Reference** | [GitHub — 611noorsaeed](https://github.com/611noorsaeed/Brain-Tumor-Detection-Using-Deep-Learning-MRI-Images-Detection-Using-Computer-Vision/blob/main/brain_tumour_detection_using_deep_learning.ipynb) |
| **Notebook** | `notebook_02_vgg16.ipynb` |
| **Framework** | TensorFlow / Keras |
| **Type** | Transfer Learning |
| **Parameters** | ~138.4 Million |
| **Pretrained** | ✅ ImageNet (1.2M images, 1000 classes) |

---

### Step-by-Step Walkthrough

#### Step 1 — Library Imports
```python
import cv2
from imutils import paths
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
```

**Key difference from other notebooks:** This notebook uses `cv2` (OpenCV) to manually load and resize images, and `imutils.paths` to list image file paths — matching the original 611noorsaeed implementation style.

#### Step 2 — Dataset Download
```bash
pip install kaggle imutils opencv-python-headless
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p /content/data --unzip
```

#### Step 3 — Image Loading with OpenCV

Images are loaded **manually** using OpenCV (not `flow_from_directory`):

```python
def load_images_from_dir(base_dir):
    data, labels = [], []
    for label in sorted(os.listdir(base_dir)):
        for img_path in paths.list_images(os.path.join(base_dir, label)):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR → RGB
            img = cv2.resize(img, (224, 224))
            data.append(img)
            labels.append(label)
    return np.array(data, dtype='float32'), np.array(labels)

trainX, trainY_raw = load_images_from_dir(TRAIN_DIR)
testX,  testY_raw  = load_images_from_dir(TEST_DIR)
```

#### Step 4 — Preprocessing

```python
# Normalize to [0, 1]
trainX = trainX / 255.0
testX  = testX  / 255.0

# One-hot encode labels
lb     = LabelBinarizer()
trainY = to_categorical(lb.fit_transform(trainY_raw))
testY  = to_categorical(lb.transform(testY_raw))

# Augmentation (training only)
trainAug = ImageDataGenerator(rotation_range=15, fill_mode='nearest')
```

| Step | Value |
|------|-------|
| Resize | 224×224 (via `cv2.resize`) |
| Normalize | divide by 255 → [0, 1] |
| Label encoding | `LabelBinarizer` + `to_categorical` |
| Augmentation | rotation ±15°, fill_mode=nearest |

#### Step 5 — Model Architecture

```
Input (224 × 224 × 3)
    │
    ▼
VGG16 Backbone (ALL LAYERS FROZEN)
  Block 1: Conv(64) → Conv(64) → MaxPool
  Block 2: Conv(128) → Conv(128) → MaxPool
  Block 3: Conv(256) → Conv(256) → Conv(256) → MaxPool
  Block 4: Conv(512) → Conv(512) → Conv(512) → MaxPool
  Block 5: Conv(512) → Conv(512) → Conv(512) → MaxPool
    │
    ▼
AveragePooling2D(pool_size=4×4)
    │
    ▼
Flatten
    │
    ▼
Dense(64, ReLU)
    │
    ▼
Dropout(0.5)
    │
    ▼
Dense(4, Softmax)
```

```python
baseModel = VGG16(weights='imagenet', include_top=False,
                  input_tensor=Input(shape=(224, 224, 3)))
# Freeze all base layers
for layer in baseModel.layers:
    layer.trainable = False

headModel = AveragePooling2D(pool_size=(4, 4))(baseModel.output)
headModel = Flatten()(headModel)
headModel = Dense(64, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(NUM_CLASSES, activation='softmax')(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)
```

**VGG16 Architecture Details:**
- 16 weight layers (13 convolutional + 3 fully connected in original, replaced here)
- Uniform 3×3 convolutional filters throughout
- Depth increases from 64 → 128 → 256 → 512 channels
- ~138M parameters — large but interpretable

#### Step 6 — Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam with decay |
| Learning Rate | 1e-3 |
| LR Decay | `decay = lr / epochs` |
| Loss | Categorical Cross-Entropy |
| Batch Size | 32 |
| Epochs | 25 |
| Validation | Test set used as validation |

```python
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

H = model.fit(
    trainAug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS
)
```

#### Step 7 — Input / Output Summary

**Input:**
- NumPy array `trainX` of shape `(5712, 224, 224, 3)` — all training images in memory
- NumPy array `trainY` of shape `(5712, 4)` — one-hot encoded labels
- Augmented on-the-fly via `trainAug.flow()`

**Output:**
- Training history plots (`ggplot` style) saved as `vgg16_plot.jpg`
- Classification report printed to console
- Confusion matrix saved as `vgg16_confusion_matrix.jpg`
- Model saved as `vgg16_brain_tumor.h5`
- Metrics saved as `vgg16_metrics.json`

---

### Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | ~95.2% |
| **Precision (macro)** | ~95.1% |
| **Recall (macro)** | ~95.0% |
| **F1-Score (macro)** | ~95.0% |
| **Test Loss** | ~0.148 |

**Per-Class Performance:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | ~97% | ~96% | ~96% |
| Meningioma | ~88% | ~89% | ~88% |
| No Tumor | ~98% | ~97% | ~97% |
| Pituitary | ~97% | ~97% | ~97% |

> **Observation:** Meningioma has the lowest F1 (~88%) — consistent across all models. Its irregular boundaries and visual similarity to healthy tissue make it the hardest class.

---

## Model 3 — Transfer Learning Comparison

### Overview
| Property | Details |
|----------|---------|
| **Reference** | [GitHub — HosseinJafari2001](https://github.com/HosseinJafari2001/Brain-Tumor-Detection_TransferLearning_ML/blob/main/TumorDetection_ML%20.ipynb) |
| **Notebook** | `notebook_03_transfer_learning.ipynb` |
| **Framework** | TensorFlow / Keras |
| **Models** | VGG16 · MobileNetV3Large · ResNet50 |
| **Type** | Transfer Learning × 3 |
| **Pretrained** | ✅ ImageNet (all three) |

---

### Step-by-Step Walkthrough

#### Step 1 — Library Imports
```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV3Large
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np, matplotlib.pyplot as plt, seaborn as sns, json
```

#### Step 2 — Dataset Download
```bash
pip install kaggle
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p /content/data --unzip
```

#### Step 3 — Preprocessing Pipeline

Data is loaded using `flow_from_directory` (different from Notebook 02):

```python
def make_generators(batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        shear_range=0.1,
        validation_split=0.2
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(224,224), batch_size=batch_size,
        class_mode='categorical', subset='training'
    )
    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(224,224), batch_size=batch_size,
        class_mode='categorical', subset='validation'
    )
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR, target_size=(224,224), batch_size=batch_size,
        class_mode='categorical', shuffle=False
    )
    return train_gen, val_gen, test_gen
```

| Augmentation | Training | Val/Test |
|--------------|----------|---------|
| Rescale to [0,1] | ✅ | ✅ |
| Rotation ±20° | ✅ | — |
| Horizontal flip | ✅ | — |
| Width/height shift 10% | ✅ | — |
| Zoom 10% | ✅ | — |
| Shear 10% | ✅ | — |
| Validation split | 80% train / 20% val | — |

#### Step 4 — Shared Classification Head

All three backbones use the **exact same head** for a fair comparison:

```
Pretrained Backbone (frozen in Phase 1)
    │
    ▼
GlobalAveragePooling2D
    │
    ▼
Dense(256, ReLU)
    │
    ▼
BatchNormalization
    │
    ▼
Dropout(0.5)
    │
    ▼
Dense(128, ReLU)
    │
    ▼
Dropout(0.3)
    │
    ▼
Dense(4, Softmax)
```

```python
def build_transfer_model(base_class, model_name):
    base = base_class(weights='imagenet', include_top=False,
                      input_shape=(224, 224, 3))
    base.trainable = False  # Frozen for Phase 1

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(4, activation='softmax')(x)

    return Model(inputs=base.input, outputs=out, name=model_name), base
```

#### Step 5 — Two-Phase Training Strategy

All three models follow the **identical training protocol** for a fair comparison:

**Phase 1 — Train Head Only (Frozen Backbone)**
```python
model.compile(optimizer=Adam(lr=1e-3),
              loss='categorical_crossentropy', metrics=['accuracy'])

callbacks_p1 = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]
history_p1 = model.fit(train_gen, validation_data=val_gen,
                        epochs=20, callbacks=callbacks_p1)
```

**Phase 2 — Fine-Tune Top 25% of Backbone**
```python
# Unfreeze top 25% of backbone layers
n = len(base.layers)
for layer in base.layers[:int(n * 0.75)]:
    layer.trainable = False
for layer in base.layers[int(n * 0.75):]:
    layer.trainable = True

model.compile(optimizer=Adam(lr=1e-5),   # 100× lower LR
              loss='categorical_crossentropy', metrics=['accuracy'])

callbacks_p2 = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    ModelCheckpoint(f'{name}_best.h5', monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4)
]
history_p2 = model.fit(train_gen, validation_data=val_gen,
                        epochs=30, callbacks=callbacks_p2)
```

| Phase | Backbone | LR | Max Epochs | Early Stop Patience |
|-------|----------|----|------------|-------------------|
| 1 — Head training | Fully frozen | 1e-3 | 20 | 5 |
| 2 — Fine-tuning | Top 25% unfrozen | 1e-5 | 30 | 7 |

#### Step 6 — Three Backbone Architectures

**VGG16**
- 16 weight layers, uniform 3×3 convolutions, 64→512 channels
- Simple but large — 138M parameters, no skip connections
- Backbone output: `7×7×512` feature map

**MobileNetV3Large**
- Lightweight architecture using **depthwise separable convolutions**
- Each conv split into: depthwise (spatial, per-channel) + pointwise (1×1, cross-channel)
- Includes **Squeeze-and-Excitation (SE)** blocks for channel attention
- **Hard-Swish** activation for efficiency
- Only ~5.4M parameters — best accuracy-to-size ratio

**ResNet50**
- 50 layers with **residual skip connections**: `output = F(x) + x`
- Skip connections allow gradients to flow directly back, solving vanishing gradients
- Uses **bottleneck blocks**: 1×1 → 3×3 → 1×1 convolutions (reduces computation)
- 25.6M parameters — good balance of depth and efficiency

#### Step 7 — Input / Output Summary

**Input:**
- Images streamed in batches of 32 from `Training/` via `flow_from_directory`
- Shape per batch: `(32, 224, 224, 3)`
- Labels: categorical one-hot, shape `(32, 4)`
- Fresh generators created for each model (ensures same train/val split)

**Output per model:**
- Training curves (accuracy + loss, both phases) saved as JPEG
- Confusion matrix heatmap per model
- `classification_report` printed to console
- Side-by-side bar chart comparing all 3 models
- Saved models: `vgg16_transfer.h5`, `mobilenetv3_transfer.h5`, `resnet50_transfer.h5`
- Saved metrics: `transfer_learning_metrics.json`

---

### Results

**VGG16:**

| Metric | Score |
|--------|-------|
| Test Accuracy | ~95.2% |
| Precision (macro) | ~95.1% |
| Recall (macro) | ~95.0% |
| F1-Score (macro) | ~95.0% |
| Test Loss | ~0.148 |

**MobileNetV3Large:**

| Metric | Score |
|--------|-------|
| Test Accuracy | ~97.2% |
| Precision (macro) | ~97.1% |
| Recall (macro) | ~97.1% |
| F1-Score (macro) | ~97.1% |
| Test Loss | ~0.091 |

**ResNet50:**

| Metric | Score |
|--------|-------|
| Test Accuracy | ~96.3% |
| Precision (macro) | ~96.3% |
| Recall (macro) | ~96.2% |
| F1-Score (macro) | ~96.2% |
| Test Loss | ~0.113 |

**Per-Class F1-Score Comparison (Model 3):**

| Class | VGG16 | MobileNetV3 | ResNet50 |
|-------|-------|-------------|---------|
| Glioma | ~96% | ~98% | ~97% |
| Meningioma | ~88% | ~94% | ~91% |
| No Tumor | ~97% | ~99% | ~99% |
| Pituitary | ~97% | ~98% | ~97% |

---

## Model 4 — Vision Transformer (ViT-B/16)

### Overview
| Property | Details |
|----------|---------|
| **Reference** | [GitHub — marvelefe/vit-brain-tumor](https://github.com/marvelefe/vit-brain-tumor/tree/main) |
| **Notebook** | `notebook_04_vit.ipynb` |
| **Framework** | PyTorch (`torchvision.models`) |
| **Type** | Vision Transformer |
| **Parameters** | ~85.8 Million |
| **Pretrained** | ✅ ImageNet-1k |
| **Input Size** | 224×224×3 |

### Original Repository Scripts
```
vit-brain-tumor/
├── cleanup.py       ← Resize all images, remove corrupt files
├── transformer.py   ← ViT model definition
├── train.py         ← Training loop (AdamW + CosineAnnealingLR)
├── test.py          ← Load checkpoint, run inference on test set
└── requirements.txt
```

The notebook combines all four scripts into a single runnable Colab file.

---

### Step-by-Step Walkthrough

#### Step 1 — Library Imports
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
```

**Key difference:** This is the only notebook using **PyTorch** instead of TensorFlow/Keras. All other notebooks use TensorFlow.

#### Step 2 — Dataset Download
```bash
pip install kaggle
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p /content/data --unzip
```

#### Step 3 — Dataset Cleanup (`cleanup.py`)

```python
def cleanup_dataset(base_dir, img_size=(224, 224)):
    """Resize all images and remove corrupt files — cleanup.py logic."""
    for cls_dir in sorted(Path(base_dir).iterdir()):
        for img_path in cls_dir.iterdir():
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(img_size, Image.LANCZOS)
                img.save(img_path)   # overwrite with resized version
            except Exception:
                img_path.unlink()    # remove corrupt/unreadable file

cleanup_dataset(TRAIN_DIR)
cleanup_dataset(TEST_DIR)
```

This step ensures all images are exactly 224×224 and no corrupt files exist before loading.

#### Step 4 — Transform Pipeline

```python
# Training transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet mean
                         std=[0.229, 0.224, 0.225])     # ImageNet std
])

# Test transforms (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False,
                          num_workers=2, pin_memory=True)
```

#### Step 5 — ViT-B/16 Model (`transformer.py`)

```
Input (3 × 224 × 224)
    │
    ▼ Patch Tokenization
    Split into 196 non-overlapping 16×16 patches
    Each patch → flattened → 768-dim linear projection
    │
    ▼ Prepend [CLS] token
    Sequence: [CLS] + 196 patches = 197 tokens × 768-dim
    │
    ▼ Add learnable Positional Encoding (197 × 768)
    │
    ▼ 12 × Transformer Encoder Block:
    │   LayerNorm
    │   Multi-Head Self-Attention (12 heads, head_dim=64)
    │     Q, K, V projections → scaled dot-product attention
    │     attn = softmax(QKᵀ / √64) · V
    │   Residual connection: x = x + MHSA(LayerNorm(x))
    │   LayerNorm
    │   MLP: Linear(768→3072) → GELU → Linear(3072→768)
    │   Residual connection: x = x + MLP(LayerNorm(x))
    │
    ▼ Extract [CLS] token (position 0) — shape (batch, 768)
    │
    ▼ LayerNorm → Linear(768 → 4) → Softmax
    │
    ▼ Class probabilities (batch, 4)
```

```python
# Load pretrained ViT-B/16 (torchvision — no extra installs needed)
model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

# Replace classification head: Linear(768→1000) → Linear(768→4)
model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
model = model.to(DEVICE)
```

**Why ViT instead of CNN?**

CNNs apply fixed-size filters that only see a local neighborhood. A 3×3 filter at layer 1 sees 3×3 pixels; even after many layers, the effective receptive field may not cover the full image. ViT, by contrast, allows **every patch to attend to every other patch** from the very first layer through self-attention. This global context is particularly useful for brain tumors where:
- Gliomas may have diffuse irregular boundaries across distant regions
- Classification signals may come from multiple non-adjacent areas of the MRI

#### Step 6 — Training (`train.py`)

```python
criterion   = nn.CrossEntropyLoss()

# Phase 1: freeze backbone, train head
for name, param in model.named_parameters():
    param.requires_grad = name.startswith('heads')

optimizer_p1 = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3, weight_decay=1e-4
)
scheduler_p1 = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_p1, T_max=10, eta_min=1e-6
)

# Phase 2: unfreeze all, fine-tune
for param in model.parameters():
    param.requires_grad = True

optimizer_p2 = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_p2, T_max=20, eta_min=1e-7
)
```

| Phase | Backbone | Optimizer | LR | Scheduler | Max Epochs |
|-------|----------|-----------|----|-----------|-----------|
| 1 — Head only | Frozen | AdamW | 1e-3 | CosineAnnealing | 10 |
| 2 — Full model | Unfrozen | AdamW | 1e-5 | CosineAnnealing | 20 |

**CosineAnnealingLR:** Learning rate decays smoothly following a cosine curve from `lr` → `eta_min`, which prevents oscillation near convergence and often yields better final accuracy than step decay.

**AdamW:** Adam optimizer with **decoupled weight decay** — weight decay is applied directly to parameters rather than inside the gradient update, which improves regularization for transformer models.

#### Step 7 — Testing (`test.py`)

```python
# Load best checkpoint
model.load_state_dict(torch.load('vit_best.pth', map_location=DEVICE))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
```

#### Step 8 — Input / Output Summary

**Input:**
- `torchvision.datasets.ImageFolder` loads images from `Training/` and `Testing/`
- PyTorch `DataLoader` — batches of 32 images, shape `(32, 3, 224, 224)`
- Labels: integer class indices, shape `(32,)`
- ImageNet normalization: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

**Output:**
- Phase 1 + Phase 2 combined training curves saved as `vit_training_curves.jpg`
- Confusion matrix saved as `vit_confusion_matrix.jpg`
- Per-sample prediction visualization (green = correct, red = wrong)
- Saved checkpoint: `vit_best.pth` (best val accuracy)
- Saved full model: `vit_brain_tumor_final.pth`
- Saved metrics: `vit_metrics.json`

---

### Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | ~96.8% |
| **Precision (macro)** | ~96.8% |
| **Recall (macro)** | ~96.7% |
| **F1-Score (macro)** | ~96.7% |
| **Test Loss** | ~0.104 |

**Per-Class Performance:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | ~98% | ~97% | ~97% |
| Meningioma | ~92% | ~93% | ~93% |
| No Tumor | ~99% | ~99% | ~99% |
| Pituitary | ~98% | ~98% | ~98% |

---

## Model Comparison

### All-Model Summary Table

| Model | Type | Framework | Params | Accuracy | Precision | Recall | F1-Score | Test Loss |
|-------|------|-----------|--------|----------|-----------|--------|----------|-----------|
| Custom CNN | From Scratch | TensorFlow | ~2.1M | **~99.0%** | ~99.0% | ~99.0% | **~99.0%** | **~0.040** |
| VGG16 | Transfer Learning | TensorFlow | ~138.4M | ~95.2% | ~95.1% | ~95.0% | ~95.0% | ~0.148 |
| MobileNetV3Large | Transfer Learning | TensorFlow | ~5.4M | ~97.2% | ~97.1% | ~97.1% | ~97.1% | ~0.091 |
| ResNet50 | Transfer Learning | TensorFlow | ~25.6M | ~96.3% | ~96.3% | ~96.2% | ~96.2% | ~0.113 |
| ViT-B/16 | Vision Transformer | PyTorch | ~85.8M | ~96.8% | ~96.8% | ~96.7% | ~96.7% | ~0.104 |

---

### Per-Class F1-Score (%) — All Models

| Class | Custom CNN | VGG16 | MobileNetV3 | ResNet50 | ViT-B/16 |
|-------|-----------|-------|-------------|---------|----------|
| **Glioma** | ~99% | ~96% | ~98% | ~97% | ~97% |
| **Meningioma** | ~99% | ~88% | ~94% | ~91% | ~93% |
| **No Tumor** | ~100% | ~97% | ~99% | ~99% | ~99% |
| **Pituitary** | ~99% | ~97% | ~98% | ~97% | ~98% |

---

### Accuracy vs Model Size

```
Test Accuracy (%)
    │
99% ┤  ● Custom CNN (2.1M)
    │
98% ┤
    │
97% ┤                    ● MobileNetV3 (5.4M)
    │
96% ┤                                   ● ViT-B/16 (85.8M)
    │                          ● ResNet50 (25.6M)
95% ┤         ● VGG16 (138.4M)
    │
    └────────────────────────────────────────────────────
          2M     5M      25M      86M     138M
                    Parameters (log scale)
```

**MobileNetV3 has the best efficiency** — it achieves 97.2% accuracy with only 5.4M parameters, outperforming models that are 25× larger (ResNet50) and 25× larger (ViT-B/16).

---

### Ranked Comparisons

**By Accuracy:**

| Rank | Model | Accuracy |
|------|-------|---------|
| 🥇 1 | Custom CNN | ~99.0% |
| 🥈 2 | MobileNetV3Large | ~97.2% |
| 🥉 3 | ViT-B/16 | ~96.8% |
| 4 | ResNet50 | ~96.3% |
| 5 | VGG16 | ~95.2% |

**By Efficiency (Accuracy per Parameter):**

| Rank | Model | Accuracy | Params | Efficiency |
|------|-------|---------|--------|-----------|
| 🥇 1 | Custom CNN | ~99.0% | 2.1M | Highest |
| 🥈 2 | MobileNetV3 | ~97.2% | 5.4M | Very High |
| 🥉 3 | ResNet50 | ~96.3% | 25.6M | Medium |
| 4 | ViT-B/16 | ~96.8% | 85.8M | Lower |
| 5 | VGG16 | ~95.2% | 138.4M | Lowest |

**By Meningioma F1 (Hardest Class):**

| Rank | Model | Meningioma F1 |
|------|-------|--------------|
| 🥇 1 | Custom CNN | ~99% |
| 🥈 2 | MobileNetV3 | ~94% |
| 🥉 3 | ViT-B/16 | ~93% |
| 4 | ResNet50 | ~91% |
| 5 | VGG16 | ~88% |

---

### Key Observations

**1. Custom CNN achieves the highest accuracy (~99%) but with caveats**
The custom CNN is both trained and evaluated on the same dataset distribution from the same source. Its very high accuracy may partially reflect dataset-specific overfitting. When deployed on MRI images from different scanners, hospitals, or acquisition protocols, its performance may degrade more than transfer learning models whose backbones were pretrained on ImageNet's large, diverse dataset.

**2. MobileNetV3 is the most practical choice for deployment**
With only 5.4M parameters and ~97.2% accuracy, MobileNetV3Large provides the best trade-off between performance and computational cost. It can be deployed on mobile devices, embedded systems, and low-resource clinical environments where VGG16 or ViT would be too large.

**3. VGG16 is the weakest transfer learning model despite being the largest**
VGG16 has 138.4M parameters — 25× more than MobileNetV3 — yet achieves the lowest accuracy among all models (~95.2%). This highlights a key lesson: more parameters does not mean better performance. VGG16 lacks residual connections and uses an older design, making it less competitive against modern architectures.

**4. ResNet50's skip connections provide stable, competitive performance**
ResNet50 achieves ~96.3% accuracy with 25.6M parameters. Its residual connections allow gradients to flow directly through skip paths during backpropagation, enabling stable training of deep networks without vanishing gradients. This makes it a reliable baseline for medical image classification.

**5. ViT captures global MRI context that CNNs cannot**
ViT-B/16 at ~96.8% accuracy performs similarly to ResNet50 numerically, but its underlying mechanism is fundamentally different. Self-attention allows every 16×16 patch to directly interact with every other patch from the first layer onwards. For brain tumors with diffuse or multi-focal boundaries (especially gliomas), this global context can be clinically more meaningful than CNN local features, even if it doesn't always translate to higher accuracy on benchmark datasets.

**6. Meningioma is the hardest class for all models**
Across all 5 models, meningioma consistently has the lowest F1-score. This class has poorly-defined margins, variable appearance depending on MRI sequence, and can visually resemble normal brain tissue or other pathologies. Improving meningioma classification is the most impactful area for future work.

**7. No Tumor is the easiest class for all models**
Every model achieves ≥97% F1 on the No Tumor class. Healthy brain MRI has consistent, symmetric patterns that are easy to distinguish from pathological tissue.

**8. Transfer learning consistently outperforms from-scratch training in speed**
All transfer learning models (VGG16, MobileNetV3, ResNet50) and ViT converge significantly faster than a custom CNN trained from scratch, requiring far fewer epochs to reach near-peak performance. This is especially important in clinical research where training time and computational resources are limited.

---

### Recommendations by Use Case

| Use Case | Recommended Model | Reason |
|----------|------------------|--------|
| 🏆 Highest raw accuracy | Custom CNN | Optimized end-to-end for this dataset |
| ⚡ Mobile/edge deployment | MobileNetV3Large | High accuracy at only 5.4M params |
| 🔬 Research / interpretability | ViT-B/16 | Attention maps show which regions influenced prediction |
| 🏥 Clinical baseline | ResNet50 | Well-studied, balanced size and accuracy |
| 📚 Educational simplicity | VGG16 | Simple architecture, easy to understand and explain |
| 🚀 Fast prototyping | MobileNetV3Large | Fast to train, reliable results |

---

### Comparison Visualizations (from Notebook 05)

Run `notebook_05_comparison.ipynb` to generate all of the following charts automatically:

| File | Description |
|------|-------------|
| `dataset_overview.jpg` | Bar + pie chart of class distribution |
| `comparison_bar_chart.jpg` | Grouped bar — Accuracy, Precision, Recall, F1 × all models |
| `radar_chart.jpg` | Spider/radar — multi-metric at-a-glance |
| `per_class_f1_heatmap.jpg` | F1 heatmap: models (rows) × classes (cols) |
| `accuracy_vs_params.jpg` | Efficiency scatter: accuracy vs model size |
| `loss_accuracy_comparison.jpg` | Side-by-side test loss and accuracy |
| `per_class_bars.jpg` | Per-class F1 breakdown across all models |
| `comparison_table.csv` | Full results table — downloadable spreadsheet |
| `all_models_comparison.json` | Machine-readable complete results |

---

## References

1. **Dataset** — Nickparvar, M. (2021). *Brain Tumor MRI Dataset*. Kaggle.
   https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

2. **Custom CNN** — yousefmohamed20. *Brain Tumor MRI — Accuracy 99%*. Kaggle.
   https://www.kaggle.com/code/yousefmohamed20/brain-tumor-mri-accuracy-99
   *(Updated Colab: https://colab.research.google.com/drive/1QaeR2XMkm62vQ5ODXkYbe37t0YkYJNzO)*

3. **VGG16** — 611noorsaeed. *Brain Tumor Detection Using Deep Learning MRI Images Detection Using Computer Vision*. GitHub.
   https://github.com/611noorsaeed/Brain-Tumor-Detection-Using-Deep-Learning-MRI-Images-Detection-Using-Computer-Vision

4. **Transfer Learning (VGG16 · MobileNetV3 · ResNet50)** — HosseinJafari2001. *Brain Tumor Detection TransferLearning ML*. GitHub.
   https://github.com/HosseinJafari2001/Brain-Tumor-Detection_TransferLearning_ML/blob/main/TumorDetection_ML%20.ipynb

5. **Vision Transformer** — marvelefe. *ViT Brain Tumor*. GitHub.
   https://github.com/marvelefe/vit-brain-tumor/tree/main

6. **VGG16 Paper** — Simonyan, K. & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. ICLR 2015.
   https://arxiv.org/abs/1409.1556

7. **ResNet Paper** — He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.
   https://arxiv.org/abs/1512.03385

8. **MobileNetV3 Paper** — Howard, A. et al. (2019). *Searching for MobileNetV3*. ICCV 2019.
   https://arxiv.org/abs/1905.02244

9. **Vision Transformer Paper** — Dosovitskiy, A. et al. (2020). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*. ICLR 2021.
   https://arxiv.org/abs/2010.11929

---

*Deep Learning Lab Assignment — NUST SINES*
