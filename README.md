# 🩺 PneumoniaMNIST AI Challenge
**Postdoctoral Technical Challenge — AlfaisalX, Alfaisal University**  
**MedX Research Unit | Medical Robotics & AI in Healthcare**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)](https://pytorch.org)
[![Colab](https://img.shields.io/badge/Open%20in-Colab-orange)](https://colab.research.google.com)

---

## 📖 Overview

An end-to-end AI system for medical image analysis using the PneumoniaMNIST dataset:

| Task | Description | Model / Tool |
|---|---|---|
| **Task 1** | CNN Classification | EfficientNet-B0 + Focal Loss |
| **Task 2** | Medical Report Generation | MedGemma-4B-IT (VLM) |
| **Task 3** | Semantic Image Retrieval | BioMedCLIP + FAISS |

---

## 🗂️ Repository Structure

```
.
├── data/
│   └── dataset.py              # Data loading, transforms, augmentation
├── models/
│   └── model.py                # Model architectures, Focal Loss
├── task1_classification/
│   ├── train.py                # Training script with configurable hyperparameters
│   └── evaluate.py             # Evaluation: metrics, confusion matrix, ROC, failures
├── task2_report_generation/
│   └── generate_reports.py     # MedGemma VLM report generation pipeline
├── task3_retrieval/
│   ├── extract_embeddings.py   # BioMedCLIP embedding extraction
│   └── search.py               # FAISS index + image/text search + evaluation
├── notebooks/
│   └── PneumoniaMNIST_Challenge_Complete.ipynb  # Google Colab notebook
├── reports/
│   ├── task1_classification_report.md
│   ├── task2_report_generation.md
│   └── task3_retrieval_system.md
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Task 1 — Train & Evaluate CNN Classifier

```bash
# Train (saves best model to outputs/task1/best_model.pth)
python task1_classification/train.py \
    --model efficientnet_b0 \
    --epochs 20 \
    --lr 1e-4 \
    --output_dir outputs/task1

# Evaluate (generates all metrics and visualizations)
python task1_classification/evaluate.py \
    --model efficientnet_b0 \
    --weights outputs/task1/best_model.pth \
    --output_dir outputs/task1
```

**Output artifacts:**
- `outputs/task1/best_model.pth` — Model weights
- `outputs/task1/training_curves.png`
- `outputs/task1/confusion_matrix.png`
- `outputs/task1/roc_curve.png`
- `outputs/task1/failure_cases.png`
- `outputs/task1/test_metrics.json`

### 3. Task 2 — Generate Medical Reports with MedGemma

```bash
# Requires HuggingFace login with MedGemma access
huggingface-cli login

# Generate reports for 10 sample images
python task2_report_generation/generate_reports.py \
    --model_id google/medgemma-4b-it \
    --num_samples 10 \
    --output_dir outputs/task2
```

> **Note**: MedGemma requires accepting terms at https://huggingface.co/google/medgemma-4b-it  
> If unavailable, the script generates mock reports with the same structure for demonstration.

### 4. Task 3 — Build & Query Retrieval System

```bash
# Step 1: Extract embeddings using BioMedCLIP
python task3_retrieval/extract_embeddings.py --output_dir outputs/task3

# Step 2: Build FAISS index
python task3_retrieval/search.py build --embeddings_dir outputs/task3

# Step 3: Image-to-image search
python task3_retrieval/search.py search --query_idx 42 --k 5 --embeddings_dir outputs/task3

# Step 4: Text-to-image search
python task3_retrieval/search.py text_search \
    --query "bilateral lung consolidation pneumonia" \
    --k 5 \
    --embeddings_dir outputs/task3

# Step 5: Evaluate Precision@k
python task3_retrieval/search.py evaluate \
    --k_values 1 5 10 \
    --n_queries 200 \
    --embeddings_dir outputs/task3
```

---

## 📊 Key Results

### Task 1 — Classification (EfficientNet-B0)

| Metric | Score |
|---|---|
| Accuracy | ~0.87 |
| AUC-ROC | ~0.94 |
| F1-Score | ~0.89 |
| Recall (Pneumonia) | ~0.87 |

### Task 2 — Report Generation (MedGemma-4B)

Three prompting strategies evaluated:
- `concise` — 2-3 sentence summary
- `structured` — Full radiology report with sections (**best quality**)
- `differential` — Differential diagnosis + conclusion

### Task 3 — Retrieval (BioMedCLIP + FAISS)

| Metric | Score |
|---|---|
| Precision@1 | ~0.87 |
| Precision@5 | ~0.83 |
| Precision@10 | ~0.81 |

---

## 🔬 Technical Design Decisions

### Why EfficientNet-B0?
- 5.3M parameters — trainable on CPU within challenge timeframe
- Compound scaling outperforms ResNet family at equivalent parameter count
- ImageNet pretraining transfers well to chest X-ray features
- Established track record on medical imaging benchmarks (CheXpert, ISIC)

### Why Focal Loss?
PneumoniaMNIST has ~74% pneumonia prevalence. Focal Loss down-weights easy well-classified examples and forces focus on hard/ambiguous cases — critical for medical classification.

### Why MedGemma?
Google's purpose-built medical VLM (2025), specifically trained on radiology images. Outperforms general-purpose VLMs (LLaVA, GPT-4V) on medical report generation benchmarks.

### Why BioMedCLIP + FAISS?
- BioMedCLIP trained on 15M PubMed biomedical figure-caption pairs — captures clinically meaningful semantics
- Enables both image-to-image AND text-to-image search from the same embedding space
- FAISS provides millisecond-latency exact search on CPU for datasets up to ~1M images

---

## 🔁 Reproducibility

All random seeds are fixed (`np.random.seed(42)`). Training is deterministic on the same hardware. To reproduce:

```bash
git clone https://github.com/YOUR_USERNAME/pneumonia-challenge
cd pneumonia-challenge
pip install -r requirements.txt
python task1_classification/train.py --epochs 20 --output_dir outputs/task1
python task1_classification/evaluate.py --output_dir outputs/task1
```

Or run the Colab notebook: `notebooks/PneumoniaMNIST_Challenge_Complete.ipynb`

---

## 📝 Reports

Detailed analysis reports for each task:
- [`reports/task1_classification_report.md`](reports/task1_classification_report.md)
- [`reports/task2_report_generation.md`](reports/task2_report_generation.md)
- [`reports/task3_retrieval_system.md`](reports/task3_retrieval_system.md)

---

## 🛠️ Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA optional (CPU sufficient for Tasks 1 & 3; recommended for Task 2)
- ~4GB RAM (8GB recommended for Task 2)

See `requirements.txt` for full dependency list.

---

## 📬 Contact

Submission for the Postdoctoral Research Fellow position at:  
**AlfaisalX: Cognitive Robotics and Autonomous Agents**  
**MedX Research Unit, Alfaisal University, Riyadh, Saudi Arabia**

Prof. Anis Koubaa: akoubaa@alfaisal.edu  
Dr. Mohamed Bahloul: mbahloul@alfaisal.edu
