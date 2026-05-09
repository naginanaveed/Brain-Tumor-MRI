# 🧠 Brain Tumor MRI Classification — Deep Learning Comparison

A comprehensive deep learning study comparing **5 models** for brain tumor classification using MRI images — from a custom CNN built from scratch to state-of-the-art Vision Transformers.

---

## 📂 Repository Structure

```
brain-tumor-mri-classification/
│
├── notebook_01_custom_cnn.ipynb         # Custom Sequential CNN (already provided)
├── notebook_02_vgg16.ipynb              # VGG16 Transfer Learning
├── notebook_03_transfer_learning.ipynb  # VGG16 + MobileNetV3Large + ResNet50
├── notebook_04_vit.ipynb                # Vision Transformer (ViT-B16)
├── notebook_05_comparison.ipynb         # ⭐ All models — unified comparison script
│
└── README.md
```

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Name** | Brain Tumor MRI Dataset |
| **Source** | [Kaggle — masoudnickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| **Total Images** | 7,023 MRI scans |
| **Format** | JPEG/PNG, Grayscale |
| **Modality** | T1, T2, and FLAIR weighted MRI |
| **Task** | 4-class classification |

### Class Distribution

| Class | Training | Testing | Total |
|-------|----------|---------|-------|
| Glioma | 1,321 | 300 | 1,621 |
| Meningioma | 1,339 | 306 | 1,645 |
| No Tumor | 1,595 | 405 | 2,000 |
| Pituitary | 1,457 | 300 | 1,757 |
| **Total** | **5,712** | **1,311** | **7,023** |

### Tumor Types Explained

- **Glioma** — Most common malignant brain tumor; arises from glial cells
- **Meningioma** — Usually benign; originates from the meninges (brain lining)
- **Pituitary** — Tumor of the pituitary gland; often benign but affects hormones
- **No Tumor** — Normal brain scan; serves as the negative class

---

## 🤖 Models

### Notebook 01 — Custom CNN (Sequential Keras)
> ⚠️ Notebook already provided — not included in this repository. Results included in comparison.

| Property | Details |
|----------|---------|
| **Type** | Built from scratch (Sequential Keras) |
| **Reference** | [Kaggle Notebook](https://www.kaggle.com/code/yousefmohamed20/brain-tumor-mri-accuracy-99) · [Colab (Updated)](https://colab.research.google.com/drive/1QaeR2XMkm62vQ5ODXkYbe37t0YkYJNzO) |
| **Architecture** | Conv2D → MaxPool → Conv2D → MaxPool → Dense |
| **Parameters** | ~2 Million |
| **Pretrained** | ❌ No |
| **Input Size** | 224×224×3 |
| **Reported Accuracy** | ~99% |

**Explanation:** A lightweight CNN designed specifically for this dataset. Layers progressively extract local features (edges → textures → patterns) through convolution and pooling operations. Fast to train with no dependency on external pretrained weights.

---

### Notebook 02 — VGG16
> 📓 `notebook_02_vgg16.ipynb`

| Property | Details |
|----------|---------|
| **Type** | Transfer Learning |
| **Reference** | [GitHub — 611noorsaeed](https://github.com/611noorsaeed/Brain-Tumor-Detection-Using-Deep-Learning-MRI-Images-Detection-Using-Computer-Vision/blob/main/brain_tumour_detection_using_deep_learning.ipynb) |
| **Architecture** | 16-layer VGG (13 Conv + 3 FC) |
| **Parameters** | ~138 Million |
| **Pretrained** | ✅ ImageNet |
| **Input Size** | 224×224×3 |

**Explanation:** VGG16 is a deep but simple architecture with uniform 3×3 convolutional filters stacked in increasing depth. The pretrained weights allow it to reuse rich visual features from ImageNet. Its large parameter count (~138M) means high capacity but slower inference.

**Tools / Libraries:**
- `tensorflow.keras.applications.VGG16`
- `ImageDataGenerator` for preprocessing and augmentation
- `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` callbacks

**Input → Output:**
- **Input:** MRI images resized to 224×224 RGB
- **Preprocessing:** Rescale to [0,1] + augmentation (rotation, flip, zoom, shear)
- **Output:** Softmax probabilities over 4 tumor classes

**Training Strategy:**
1. Freeze VGG16 base → train custom head (lr = 1e-3)
2. Unfreeze last 4 conv layers → fine-tune (lr = 1e-5)

---

### Notebook 03 — Transfer Learning Comparison (VGG16 + MobileNetV3 + ResNet50)
> 📓 `notebook_03_transfer_learning.ipynb`

| Property | Details |
|----------|---------|
| **Type** | Transfer Learning × 3 models |
| **Reference** | [GitHub — HosseinJafari2001](https://github.com/HosseinJafari2001/Brain-Tumor-Detection_TransferLearning_ML/blob/main/TumorDetection_ML%20.ipynb) |
| **Models** | VGG16, MobileNetV3Large, ResNet50 |
| **Pretrained** | ✅ ImageNet (all three) |
| **Input Size** | 224×224×3 |

**Explanation:**

| Model | Design Idea | Params | Strength |
|-------|-------------|--------|---------|
| VGG16 | Sequential 3×3 convolutions | ~138M | High accuracy, proven |
| MobileNetV3Large | Depthwise separable + SE blocks | ~5.4M | Efficient, mobile-friendly |
| ResNet50 | Residual skip connections | ~25M | Avoids vanishing gradients |

**Tools / Libraries:**
- `tensorflow.keras.applications.{VGG16, MobileNetV3Large, ResNet50}`
- `sklearn.metrics` for classification_report, confusion_matrix
- `seaborn` for heatmap visualization

**Input → Output:**
- **Input:** MRI images 224×224 (same for all three)
- **Output:** Per-model classification report + side-by-side confusion matrices + metric comparison bar chart

**Training Strategy:** Identical 2-phase protocol applied to all three for fair comparison:
1. Freeze base → train head (Adam lr = 1e-3, 15 epochs max)
2. Unfreeze top 25% → fine-tune (Adam lr = 1e-5, 25 epochs max)

---

### Notebook 04 — Vision Transformer (ViT-B16)
> 📓 `notebook_04_vit.ipynb`

| Property | Details |
|----------|---------|
| **Type** | Vision Transformer |
| **Reference** | [GitHub — marvelefe/vit-brain-tumor](https://github.com/marvelefe/vit-brain-tumor/tree/main) |
| **Architecture** | ViT-B16 (12 Transformer blocks, 768 hidden dim) |
| **Parameters** | ~86 Million |
| **Pretrained** | ✅ ImageNet-21k |
| **Input Size** | 224×224×3 (split into 196 patches of 16×16) |

**Explanation:** Vision Transformers treat an image as a sequence of patches rather than a grid. Each 16×16 patch is embedded into a 768-dim token. A [CLS] token is prepended, positional encodings are added, and 12 Multi-Head Self-Attention (MHSA) blocks process the sequence. The final [CLS] token is used for classification. Unlike CNNs, ViTs have **no inductive bias** (no assumption of locality), so they rely on large-scale pretraining to learn spatial structure.

**Tools / Libraries:**
- `vit-keras` — pip installable ViT implementation for TensorFlow
- `tensorflow.keras` — model building, training loops
- `CosineDecay` learning rate schedule for fine-tuning
- `sklearn.metrics` for evaluation

**Input → Output:**
- **Input:** 224×224 RGB MRI image → split into 196 patches of 16×16 pixels
- **Patch Embedding:** Each patch → 768-dimensional vector
- **Transformer Processing:** 12 blocks of Multi-Head Self-Attention + MLP
- **Output:** [CLS] token → Dense(256) → Softmax(4 classes)

**Training Strategy:**
1. Freeze ViT backbone → train 2-layer head (lr = 1e-3, 10 epochs)
2. Unfreeze all layers → fine-tune with Cosine Decay schedule (lr = 1e-5 → 1e-7, 30 epochs)

---

### Notebook 05 — All-Model Comparison ⭐
> 📓 `notebook_05_comparison.ipynb` — *Required for the 10-grade assignment*

This single notebook trains all 4 models (VGG16, MobileNetV3, ResNet50, ViT-B16) from scratch on the same dataset split and loads Custom CNN metrics for a fair, unified comparison.

**Outputs generated:**
- Comparison summary table (accuracy, precision, recall, F1, loss, params)
- Grouped bar chart: all metrics side-by-side
- Radar/spider chart: multi-metric visual comparison
- Confusion matrices (2×2 grid) with count + percentage
- Accuracy vs Parameters scatter plot (efficiency analysis)
- `comparison_results.json` and `comparison_table.csv` exports

---

## 📈 Model Comparison

> *Results shown are typical values reported in literature for this dataset. Your actual results may vary slightly depending on random seed, GPU, and training duration.*

| Model | Type | Params | Accuracy | Precision | Recall | F1-Score | Test Loss |
|-------|------|--------|----------|-----------|--------|----------|-----------|
| Custom CNN | From Scratch | ~2M | ~99.0% | ~99.0% | ~99.0% | ~99.0% | ~0.04 |
| VGG16 | Transfer Learning | ~138M | ~96–98% | ~96–98% | ~96–98% | ~96–98% | ~0.10 |
| MobileNetV3Large | Transfer Learning | ~5.4M | ~97–99% | ~97–99% | ~97–99% | ~97–99% | ~0.07 |
| ResNet50 | Transfer Learning | ~25M | ~95–99% | ~95–99% | ~95–98% | ~95–98% | ~0.08 |
| ViT-B16 | Vision Transformer | ~86M | ~96–98% | ~96–98% | ~96–98% | ~96–98% | ~0.09 |

### 🔍 Key Observations

1. **Custom CNN achieves surprisingly high accuracy (~99%)** — likely due to dataset-specific overfitting or the model being tuned specifically for this task. On new/unseen hospital data, generalization may be limited.

2. **MobileNetV3Large offers the best accuracy-to-parameter ratio** — achieves competitive accuracy with only ~5.4M parameters, making it ideal for mobile/edge deployment.

3. **VGG16 is the largest model** (~138M params) but does not proportionally outperform lighter alternatives, due to its lack of residual connections and modern design choices.

4. **ResNet50's skip connections** solve the vanishing gradient problem, enabling stable training of deeper layers and competitive performance.

5. **ViT-B16 captures global spatial context** via self-attention, which can be advantageous for finding diffuse or irregular tumor boundaries. However, it needs strong pretraining (ImageNet-21k) to match CNN performance on small-medium datasets.

6. **All transfer learning models outperform from-scratch training** in terms of training stability and convergence speed, requiring fewer epochs to reach peak accuracy.

### 🏆 Model Ranking (by efficiency + accuracy)

| Rank | Model | Reason |
|------|-------|--------|
| 🥇 1st | MobileNetV3Large | High accuracy, smallest parameter count |
| 🥈 2nd | ResNet50 | Strong accuracy, residual design, moderate size |
| 🥉 3rd | ViT-B16 | Global attention, strong generalization |
| 4th | Custom CNN | Best raw accuracy but may overfit |
| 5th | VGG16 | Large and older architecture; outclassed by modern alternatives |

---

## 🛠️ Tools & Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| TensorFlow | ≥ 2.10 | Deep learning framework |
| Keras | (via TF) | Model building API |
| vit-keras | latest | ViT-B16 implementation for TensorFlow |
| scikit-learn | ≥ 1.0 | Metrics: classification_report, confusion_matrix |
| NumPy | ≥ 1.21 | Array operations |
| Matplotlib | ≥ 3.5 | Plotting training curves, comparisons |
| Seaborn | ≥ 0.11 | Confusion matrix heatmaps |
| Pandas | ≥ 1.3 | Results table handling |
| kaggle | latest | Dataset download via API |

### Installation

```bash
pip install tensorflow vit-keras scikit-learn matplotlib seaborn pandas kaggle
```

---

## 🚀 How to Run

### Prerequisites
1. A Google Colab or local environment with GPU
2. Kaggle API key (`kaggle.json`) — download from [kaggle.com/account](https://www.kaggle.com/account)

### Steps
```
1. Open any notebook in Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run Cell 1 (installs dependencies)
4. Upload kaggle.json when prompted
5. Run remaining cells in order
```

### For the Comparison Notebook (Notebook 05)
```
1. Open notebook_05_comparison.ipynb in Colab
2. Enable T4 GPU
3. Run all cells — trains all 4 models sequentially (~60–90 min on T4)
4. For Custom CNN, either upload custom_cnn_metrics.json or edit the manual values
```

---

## 📐 Preprocessing Pipeline (All Models)

```
Raw MRI Image (variable size)
        │
        ▼
   Resize to 224×224
        │
        ▼
  Normalize to [0, 1]
        │
        ├──── Training ────▶  Augmentation:
        │                     • Rotation ±15°
        │                     • Horizontal flip
        │                     • Zoom 10%
        │                     • Width/height shift 10%
        │                     • Shear 10%
        │
        └──── Val/Test ────▶  No augmentation
                              (only rescale)
        │
        ▼
  ImageDataGenerator → Batches of 32 (16 for ViT)
```

---

## 📋 Training Configuration

| Setting | CNN Transfer Models | ViT-B16 |
|---------|---------------------|---------|
| Image Size | 224×224 | 224×224 |
| Batch Size | 32 | 16 |
| Phase 1 LR | 1e-3 | 1e-3 |
| Phase 2 LR | 1e-5 | CosineDecay (1e-5 → 1e-7) |
| Optimizer | Adam | Adam |
| Loss | Categorical Crossentropy | Categorical Crossentropy |
| Val Split | 20% of training set | 20% of training set |
| Early Stopping | Patience = 5 (P1), 7 (P2) | Patience = 4 (P1), 8 (P2) |

---

## 📚 References

1. **Dataset:** Nickparvar, M. (2021). Brain Tumor MRI Dataset. Kaggle. https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

2. **Custom CNN:** yousefmohamed20. Brain Tumor MRI — Accuracy 99%. https://www.kaggle.com/code/yousefmohamed20/brain-tumor-mri-accuracy-99

3. **VGG16 Reference:** 611noorsaeed. Brain Tumor Detection Using VGG16. https://github.com/611noorsaeed/Brain-Tumor-Detection-Using-Deep-Learning-MRI-Images-Detection-Using-Computer-Vision

4. **Transfer Learning Reference:** HosseinJafari2001. Brain Tumor Detection — Transfer Learning. https://github.com/HosseinJafari2001/Brain-Tumor-Detection_TransferLearning_ML

5. **ViT Reference:** marvelefe. ViT Brain Tumor. https://github.com/marvelefe/vit-brain-tumor

6. **VGG16 Architecture:** Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR 2015.

7. **ResNet50 Architecture:** He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.

8. **MobileNetV3:** Howard, A., et al. (2019). Searching for MobileNetV3. ICCV 2019.

9. **Vision Transformer:** Dosovitskiy, A., et al. (2020). An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale. ICLR 2021.

---

*Developed for the Deep Learning Lab Assignment — NUST SINES*
