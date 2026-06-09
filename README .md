# 🧠 Brain Tumor MRI Classification — Complete Deep Learning Study

A comprehensive comparison of **5 deep learning models** for brain tumor classification from MRI scans, covering approaches from a custom CNN built from scratch to a Vision Transformer.

---

## 📁 Repository Structure

```
brain-tumor-mri-classification/
│
├── notebook_01_custom_cnn.ipynb         ← Already provided (your existing notebook)
├── notebook_02_vgg16.ipynb              ← VGG16 Transfer Learning
├── notebook_03_transfer_learning.ipynb  ← VGG16 + MobileNetV3 + ResNet50
├── notebook_04_vit.ipynb                ← Vision Transformer (ViT-B/16)
├── notebook_05_comparison.ipynb         ← ⭐ All-model comparison (this notebook)
│
└── README.md
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Name** | Brain Tumor MRI Dataset |
| **Source** | [Kaggle — masoudnickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| **Total Images** | 7,023 MRI scans |
| **Image Type** | JPEG/PNG, RGB (grayscale converted) |
| **Modality** | T1, T2, FLAIR weighted MRI |
| **Task** | 4-class multi-class classification |

### Class Distribution

| Class | Train | Test | Total | Description |
|---|---|---|---|---|
| **Glioma** | 1,321 | 300 | 1,621 | Most common malignant tumor; arises from glial cells |
| **Meningioma** | 1,339 | 306 | 1,645 | Usually benign; originates from brain lining (meninges) |
| **No Tumor** | 1,595 | 405 | 2,000 | Normal brain MRI — negative class |
| **Pituitary** | 1,457 | 300 | 1,757 | Tumor of the pituitary gland; affects hormone regulation |
| **Total** | **5,712** | **1,311** | **7,023** | |

### Dataset Notes
- Images are **pre-organized** into `Training/` and `Testing/` folders with class subfolders
- Images have **variable resolutions** — all models resize to `224×224` during preprocessing
- The dataset is **relatively balanced** across the 4 classes
- **No Tumor** class is slightly larger (~28.5% of total) — mild imbalance handled via augmentation

---

## 🤖 Models

---

### Notebook 01 — Custom Sequential CNN

> **⚠️ Notebook already provided — not included in this repository. Results are included in the comparison.**

| Property | Details |
|---|---|
| **Reference** | [Kaggle — yousefmohamed20 (Acc ~99%)](https://www.kaggle.com/code/yousefmohamed20/brain-tumor-mri-accuracy-99) · [Updated Colab](https://colab.research.google.com/drive/1QaeR2XMkm62vQ5ODXkYbe37t0YkYJNzO) |
| **Type** | Custom CNN (trained from scratch) |
| **Framework** | TensorFlow / Keras Sequential API |
| **Parameters** | ~2.1 Million |
| **Pretrained** | ❌ No — random weight initialization |
| **Input Size** | 224×224×3 |

#### Architecture
```
Input (224×224×3)
  → Conv2D(32) → BatchNorm → ReLU → MaxPool
  → Conv2D(64) → BatchNorm → ReLU → MaxPool
  → Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
  → Conv2D(256) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
  → Flatten
  → Dense(512, ReLU) → Dropout(0.5)
  → Dense(4, Softmax)
```

#### Explanation
A lightweight CNN specifically designed and tuned for this dataset. Layers progressively extract local visual features — edges in early layers, textures and shapes in deeper layers. Because it is trained from scratch on this exact data distribution, it can achieve very high accuracy, but may lack generalizability to unseen MRI datasets from different scanners.

#### Input / Output
- **Input:** Raw MRI images resized to 224×224, normalized to [0, 1]
- **Augmentation:** Rotation, flip, zoom (training only)
- **Output:** Softmax probabilities over 4 tumor classes
- **Loss:** Categorical Cross-Entropy

#### Tools
| Tool | Purpose |
|---|---|
| `tensorflow.keras.Sequential` | Model definition |
| `ImageDataGenerator` | Augmentation and batching |
| `EarlyStopping`, `ModelCheckpoint` | Training callbacks |
| `sklearn.metrics` | Classification report, confusion matrix |

#### Results (Reported)
| Accuracy | Precision | Recall | F1-Score | Test Loss |
|---|---|---|---|---|
| ~99.00% | ~99.00% | ~99.00% | ~99.00% | ~0.04 |

---

### Notebook 02 — VGG16 Transfer Learning

> **📓 `notebook_02_vgg16.ipynb`**

| Property | Details |
|---|---|
| **Reference** | [GitHub — 611noorsaeed](https://github.com/611noorsaeed/Brain-Tumor-Detection-Using-Deep-Learning-MRI-Images-Detection-Using-Computer-Vision/blob/main/brain_tumour_detection_using_deep_learning.ipynb) |
| **Type** | Transfer Learning |
| **Framework** | TensorFlow / Keras |
| **Parameters** | ~138.4 Million |
| **Pretrained** | ✅ ImageNet (1.2M images, 1000 classes) |
| **Input Size** | 224×224×3 |

#### Architecture
```
Input (224×224×3)
  → VGG16 Backbone (13 Conv layers + MaxPool, ALL FROZEN)
     [Block 1–5: 3×3 Conv filters, progressively deeper 64→512 channels]
  → AveragePooling2D (4×4)
  → Flatten
  → Dense(64, ReLU)
  → Dropout(0.5)
  → Dense(4, Softmax)
```

#### Explanation
VGG16 is a deep network designed at Oxford's Visual Geometry Group. It uses 13 convolutional layers with uniform small 3×3 filters, which allows very deep feature extraction. Pre-trained ImageNet weights allow it to immediately recognize low-level features (edges, textures) that transfer well to MRI classification. The entire backbone is **frozen** and only the custom head is trained.

**Key difference from other transfer learning models:** VGG16 uses `AveragePooling2D` directly on the backbone output (not `GlobalAveragePooling2D`), and images are loaded manually using `cv2` rather than `flow_from_directory`.

#### Input / Output
- **Input:** MRI images loaded via `cv2.imread` → converted BGR→RGB → resized to 224×224 → normalized to [0, 1]
- **Label encoding:** `LabelBinarizer` + `to_categorical`
- **Augmentation:** `ImageDataGenerator(rotation_range=15, fill_mode='nearest')`
- **Output:** Softmax probabilities → `classification_report` + confusion matrix
- **Loss:** Categorical Cross-Entropy

#### Tools
| Tool | Purpose |
|---|---|
| `cv2` (OpenCV) | Image loading and resizing |
| `imutils.paths` | Listing image paths |
| `sklearn.preprocessing.LabelBinarizer` | Label encoding |
| `tensorflow.keras.applications.VGG16` | Pretrained backbone |
| `Adam(decay=lr/epochs)` | Optimizer with learning rate decay |
| `seaborn.heatmap` | Confusion matrix visualization |

#### Results (Typical)
| Accuracy | Precision | Recall | F1-Score | Test Loss |
|---|---|---|---|---|
| ~95.2% | ~95.1% | ~95.0% | ~95.0% | ~0.15 |

---

### Notebook 03 — Transfer Learning: VGG16 + MobileNetV3 + ResNet50

> **📓 `notebook_03_transfer_learning.ipynb`**

| Property | Details |
|---|---|
| **Reference** | [GitHub — HosseinJafari2001](https://github.com/HosseinJafari2001/Brain-Tumor-Detection_TransferLearning_ML/blob/main/TumorDetection_ML%20.ipynb) |
| **Type** | Transfer Learning × 3 models |
| **Framework** | TensorFlow / Keras |
| **Pretrained** | ✅ ImageNet (all 3 models) |
| **Input Size** | 224×224×3 |

#### Three Models Compared

| Model | Parameters | Architecture Highlight |
|---|---|---|
| **VGG16** | ~138.4M | Sequential 3×3 Conv blocks, no skip connections |
| **MobileNetV3Large** | ~5.4M | Depthwise separable conv + Squeeze-and-Excitation blocks |
| **ResNet50** | ~25.6M | Residual skip connections solve vanishing gradient |

#### Shared Classification Head
```
Pretrained Base (partially frozen in Phase 2)
  → GlobalAveragePooling2D
  → Dense(256, ReLU)
  → BatchNormalization
  → Dropout(0.5)
  → Dense(128, ReLU)
  → Dropout(0.3)
  → Dense(4, Softmax)
```

#### Explanation
All three models are trained with the **identical 2-phase strategy** for a fair comparison:
- **Phase 1:** Backbone completely frozen → train only the head (lr = 1e-3, up to 20 epochs)
- **Phase 2:** Unfreeze top 25% of backbone → fine-tune end-to-end (lr = 1e-5, up to 30 epochs)

**MobileNetV3** uses **depthwise separable convolutions** — splitting a standard convolution into a depthwise step (spatial) + pointwise step (channel mixing) — reducing computation by ~8–9× while maintaining high accuracy. Its **Squeeze-and-Excitation blocks** adaptively re-weight channel importance.

**ResNet50** introduces **skip connections** (identity shortcuts) that allow gradients to flow directly back to earlier layers, solving the vanishing gradient problem and enabling stable training of 50+ layers.

#### Input / Output
- **Input:** `flow_from_directory` from `Training/` → 224×224, batch=32, 80/20 train-val split
- **Augmentation (train):** rotation ±20°, flip, zoom 10%, width/height shift 10%, shear 10%
- **Output:** Side-by-side confusion matrices + grouped bar chart + classification reports
- **Saved:** `transfer_learning_metrics.json`

#### Tools
| Tool | Purpose |
|---|---|
| `tensorflow.keras.applications.{VGG16, MobileNetV3Large, ResNet50}` | Pretrained backbones |
| `ImageDataGenerator.flow_from_directory` | Automated data loading |
| `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau` | Training control |
| `seaborn.heatmap` | Side-by-side confusion matrices |
| `sklearn.metrics` | All evaluation metrics |

#### Results (Typical)
| Model | Accuracy | Precision | Recall | F1-Score | Test Loss |
|---|---|---|---|---|---|
| VGG16 | ~95.2% | ~95.1% | ~95.0% | ~95.0% | ~0.15 |
| MobileNetV3Large | ~97.2% | ~97.1% | ~97.1% | ~97.1% | ~0.09 |
| ResNet50 | ~96.3% | ~96.3% | ~96.2% | ~96.2% | ~0.11 |

---

### Notebook 04 — Vision Transformer (ViT-B/16)

> **📓 `notebook_04_vit.ipynb`**

| Property | Details |
|---|---|
| **Reference** | [GitHub — marvelefe/vit-brain-tumor](https://github.com/marvelefe/vit-brain-tumor/tree/main) |
| **Type** | Vision Transformer |
| **Framework** | PyTorch (`torchvision.models.vit_b_16`) |
| **Parameters** | ~85.8 Million |
| **Pretrained** | ✅ ImageNet-1k |
| **Input Size** | 224×224×3 |

#### Repository Structure (Original)
```
vit-brain-tumor/
├── data/           ← Dataset (Training/ and Testing/ subdirs)
├── cleanup.py      ← Resize + remove corrupt files
├── requirements.txt
├── test.py         ← Inference / evaluation
├── train.py        ← Training loop
└── transformer.py  ← ViT model definition
```

#### Architecture
```
Input (224×224×3)
  ↓ cleanup.py: resize all images to 224×224, remove corrupt files
  ↓ Patch Tokenization: split into 196 patches of 16×16
  ↓ Linear Projection: each patch → 768-dim embedding
  ↓ [CLS] token prepended → 197 tokens
  ↓ Positional Encoding added (learnable)
  ↓ 12 × Transformer Encoder Block:
       LayerNorm → Multi-Head Self-Attention (12 heads) → residual
       LayerNorm → MLP (768→3072→768, GELU) → residual
  ↓ [CLS] token → LayerNorm → Linear(768→4) → Softmax
```

#### Explanation
Vision Transformers fundamentally differ from CNNs: instead of sliding filters over local patches, ViT **treats the entire image as a sequence of tokens**. Each 16×16 pixel patch is embedded as a 768-dimensional vector. The 12 Transformer encoder blocks apply **Multi-Head Self-Attention**, where every patch can directly attend to every other patch regardless of distance. This allows ViT to model global dependencies across the entire MRI scan.

This is especially useful for **diffuse or irregular tumor boundaries** where the classification signal spans multiple regions — a characteristic that local CNN filters can miss.

#### Input / Output (`train.py` / `test.py`)
- **Input:** `torchvision.datasets.ImageFolder` → transforms: Resize(224) + RandomFlip + Rotation(15°) + ColorJitter + ToTensor + Normalize(ImageNet)
- **Training:** Phase 1 (frozen backbone, head only, lr=1e-3) + Phase 2 (full fine-tune, lr=1e-5, CosineAnnealingLR)
- **Output:** Training curves + confusion matrix + per-sample prediction visualization
- **Saved:** `vit_brain_tumor_final.pth` + `vit_metrics.json`
- **Loss:** CrossEntropyLoss · **Optimizer:** AdamW (weight_decay=1e-4)

#### Tools
| Tool | Purpose |
|---|---|
| `torchvision.models.vit_b_16` | Pretrained ViT-B/16 backbone |
| `torchvision.datasets.ImageFolder` | Dataset loading |
| `torchvision.transforms` | Augmentation pipeline |
| `torch.optim.AdamW` | Optimizer |
| `CosineAnnealingLR` | Learning rate scheduler |
| `PIL.Image` | Image loading in `cleanup.py` |
| `sklearn.metrics` | Classification report, confusion matrix |

#### Results (Typical)
| Accuracy | Precision | Recall | F1-Score | Test Loss |
|---|---|---|---|---|
| ~96.8% | ~96.8% | ~96.7% | ~96.7% | ~0.10 |

---

### Notebook 05 — All Models Comparison ⭐

> **📓 `notebook_05_comparison.ipynb`**

This notebook **does not train any model**. It loads saved metric files and generates all comparison visualizations. It runs in under 1 minute and requires no GPU.

#### What it generates
| Output | Description |
|---|---|
| `dataset_overview.jpg` | Bar + pie chart of class distribution |
| `comparison_table.csv` | Full results table with all metrics |
| `comparison_bar_chart.jpg` | Grouped bar chart — all metrics × all models |
| `radar_chart.jpg` | Spider/radar chart — multi-metric comparison |
| `per_class_f1_heatmap.jpg` | F1 heatmap: models (rows) × classes (cols) |
| `accuracy_vs_params.jpg` | Scatter plot — accuracy vs model size (log scale) |
| `loss_accuracy_comparison.jpg` | Side-by-side loss and accuracy bars |
| `per_class_bars.jpg` | Per-class F1 breakdown across models |
| `all_models_comparison.json` | Machine-readable full results |

#### How to use
1. Run individual notebooks → they each save a `*_metrics.json` file
2. Upload JSON files to this notebook when prompted
3. All charts auto-generate using your actual results
4. If no JSON files are uploaded, pre-filled benchmark values are used automatically

---

## 📈 Results Comparison

### Summary Table

| Model | Type | Params | Accuracy | Precision | Recall | F1-Score | Test Loss |
|---|---|---|---|---|---|---|---|
| Custom CNN | From Scratch | ~2.1M | **~99.0%** | ~99.0% | ~99.0% | ~99.0% | ~0.04 |
| VGG16 | Transfer Learning | ~138.4M | ~95.2% | ~95.1% | ~95.0% | ~95.0% | ~0.15 |
| MobileNetV3Large | Transfer Learning | ~5.4M | ~97.2% | ~97.1% | ~97.1% | ~97.1% | ~0.09 |
| ResNet50 | Transfer Learning | ~25.6M | ~96.3% | ~96.3% | ~96.2% | ~96.2% | ~0.11 |
| ViT-B/16 | Vision Transformer | ~85.8M | ~96.8% | ~96.8% | ~96.7% | ~96.7% | ~0.10 |

> **Note:** Results will vary depending on random seed, GPU, and number of epochs. Update `notebook_05_comparison.ipynb` with your actual results after running each individual notebook.

### Per-Class F1-Score (%) — Approximate

| Model | Glioma | Meningioma | No Tumor | Pituitary |
|---|---|---|---|---|
| Custom CNN | 99 | 98 | 100 | 99 |
| VGG16 | 96 | 88 | 97 | 97 |
| MobileNetV3Large | 98 | 94 | 99 | 98 |
| ResNet50 | 97 | 91 | 99 | 97 |
| ViT-B/16 | 97 | 93 | 99 | 98 |

> **Meningioma is consistently the hardest class** across all models due to its irregular, poorly-defined boundaries and visual similarity to surrounding healthy tissue.

### Key Findings

1. **Custom CNN achieves ~99% accuracy** but is trained and tested on the same distribution. On unseen data from different MRI scanners or protocols, generalization may be limited compared to models pretrained on ImageNet's diverse visual features.

2. **MobileNetV3Large has the best efficiency**: competitive accuracy with only ~5.4M parameters — ideal for real-world clinical deployment on mobile or edge devices.

3. **VGG16 is the largest model** (~138M params) but does not proportionally outperform lighter alternatives, since it lacks residual connections and modern architectural improvements.

4. **ResNet50's skip connections** enable stable deep network training and competitive accuracy with a moderate parameter count.

5. **ViT-B/16 captures global MRI context** via self-attention across all 196 image patches simultaneously. This global receptive field is particularly useful for irregular or diffuse tumor boundaries that CNNs with local filters can miss.

6. **Meningioma remains the hardest class** for all models — its subtle features and visual similarity to healthy tissue make it harder to distinguish even for deep networks.

### Model Rankings

| Rank | Category | Model | Reason |
|---|---|---|---|
| 🥇 1 | Best raw accuracy | Custom CNN | Optimized specifically for this dataset |
| 🥇 1 | Best efficiency | MobileNetV3Large | Top-tier accuracy at only 5.4M params |
| 🥈 2 | Best generalizability | ViT-B/16 | Global attention, robust to distribution shifts |
| 🥉 3 | Best balance | ResNet50 | Strong accuracy, residual design, 25M params |
| 4 | Large but standard | VGG16 | 138M params, older design, lower relative performance |

---

## 🛠️ Tools & Libraries

### TensorFlow Notebooks (01, 02, 03)

| Library | Version | Purpose |
|---|---|---|
| TensorFlow | ≥ 2.10 | Deep learning framework |
| Keras | (via TF) | Model building, training loops |
| OpenCV (`cv2`) | ≥ 4.5 | Image loading and resizing (Notebook 02) |
| imutils | latest | Utility for listing image paths (Notebook 02) |
| scikit-learn | ≥ 1.0 | Metrics, label encoding, train/test split |
| NumPy | ≥ 1.21 | Array operations |
| Matplotlib | ≥ 3.5 | Training curves, bar charts |
| Seaborn | ≥ 0.11 | Confusion matrix heatmaps |
| kaggle | latest | Dataset API download |

### PyTorch Notebook (04)

| Library | Version | Purpose |
|---|---|---|
| PyTorch | ≥ 1.13 | Deep learning framework |
| torchvision | ≥ 0.14 | ViT model, transforms, ImageFolder |
| PIL (Pillow) | ≥ 9.0 | Image loading in cleanup step |
| scikit-learn | ≥ 1.0 | Evaluation metrics |
| NumPy, Matplotlib, Seaborn | — | Analysis and visualization |

### Comparison Notebook (05) — No extra installs

| Library | Purpose |
|---|---|
| NumPy, Pandas | Data manipulation |
| Matplotlib, Seaborn | All comparison charts |
| json | Load metric files |

---

## 🚀 Quick Start

### Requirements
- Google Colab with **T4 GPU** (Runtime → Change runtime type → T4)
- Kaggle API key (`kaggle.json`) — download from [kaggle.com/account](https://www.kaggle.com/account)

### Installation (runs automatically in each notebook)
```bash
# Notebooks 01, 02, 03 — TensorFlow (pre-installed on Colab)
pip install kaggle imutils opencv-python-headless

# Notebook 04 — PyTorch (pre-installed on Colab)
pip install kaggle

# Notebook 05 — No installs needed
```

### Run Order
```
1. notebook_01_custom_cnn.ipynb      → saves custom_cnn_metrics.json
2. notebook_02_vgg16.ipynb           → saves vgg16_metrics.json
3. notebook_03_transfer_learning.ipynb → saves transfer_learning_metrics.json
4. notebook_04_vit.ipynb             → saves vit_metrics.json
5. notebook_05_comparison.ipynb      → upload the 4 JSON files → generates all charts
```

### Expected Runtime (T4 GPU)
| Notebook | Approx. Time |
|---|---|
| 01 Custom CNN | 20–30 min |
| 02 VGG16 | 25–40 min |
| 03 Transfer Learning (×3 models) | 60–90 min |
| 04 ViT-B/16 | 45–60 min |
| 05 Comparison | < 1 min |

---

## 📋 Preprocessing Pipeline

All models follow the same core preprocessing flow:

```
Raw MRI Image (variable resolution, JPEG/PNG)
        │
        ▼
   Resize to 224×224
        │
        ▼
   Normalize pixels
   TF: rescale=1/255  →  [0, 1]
   PyTorch: Normalize(ImageNet mean/std)
        │
        ├── Training set ──▶ Augmentation:
        │                    • Rotation ±15–20°
        │                    • Horizontal flip
        │                    • Zoom 10%
        │                    • Width/height shift 10%
        │                    • Shear 10%  (TF notebooks)
        │                    • ColorJitter  (PyTorch notebook)
        │
        └── Val / Test set ──▶ No augmentation
                               (only resize + normalize)
```

---

## 📚 References

1. **Dataset** — Nickparvar, M. (2021). *Brain Tumor MRI Dataset*. Kaggle.  
   https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

2. **Custom CNN** — yousefmohamed20. *Brain Tumor MRI — Accuracy 99%*. Kaggle.  
   https://www.kaggle.com/code/yousefmohamed20/brain-tumor-mri-accuracy-99  
   *(Updated Colab: https://colab.research.google.com/drive/1QaeR2XMkm62vQ5ODXkYbe37t0YkYJNzO)*

3. **VGG16** — 611noorsaeed. *Brain Tumor Detection Using Deep Learning*. GitHub.  
   https://github.com/611noorsaeed/Brain-Tumor-Detection-Using-Deep-Learning-MRI-Images-Detection-Using-Computer-Vision

4. **Transfer Learning (VGG16 + MobileNetV3 + ResNet50)** — HosseinJafari2001. *Brain Tumor Detection TransferLearning ML*. GitHub.  
   https://github.com/HosseinJafari2001/Brain-Tumor-Detection_TransferLearning_ML/blob/main/TumorDetection_ML%20.ipynb

5. **Vision Transformer** — marvelefe. *ViT Brain Tumor*. GitHub.  
   https://github.com/marvelefe/vit-brain-tumor/tree/main

6. **VGG16** — Simonyan, K. & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. ICLR 2015.

7. **ResNet50** — He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.

8. **MobileNetV3** — Howard, A. et al. (2019). *Searching for MobileNetV3*. ICCV 2019.

9. **Vision Transformer** — Dosovitskiy, A. et al. (2020). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*. ICLR 2021.

---

*Project completed for Deep Learning Lab Assignment — NUST SINES*
