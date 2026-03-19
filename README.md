# Glimpse

**Comparing Multimodal Fusion Strategies for Visual Question Answering**

A deep learning research project that investigates how vision and language representations should be combined to answer natural language questions about images. Built from scratch as a semester project at IISER Bhopal.

---

## The Research Question

> *Given a frozen vision encoder and a frozen text encoder, which fusion strategy produces the best yes/no answer predictions — and does LoRA adaptation of the encoder change which fusion wins?*

---

## What is VQA?

Visual Question Answering (VQA) is the task of answering a natural language question about an image.

```
Image: a photo of a park
Question: "Is there a dog in the image?"
Answer: Yes
```

We use **VQA-v2** — a dataset of 214,354 question-answer pairs over 40,504 real images (COCO val2014). We filter to yes/no questions only (~80,537 samples) to keep the output binary and isolate the fusion comparison.

**Why VQA-v2 and not v1?** V1 had a language bias — models could guess correctly without looking at the image. V2 fixes this by ensuring every question has a complementary image where the answer flips.

---

## System Architecture

Every sample flows through three stages:

```
Image    →  [CLIP Vision Encoder]  →  image_feat (512d)
                                                          →  [Fusion]  →  [Classifier]  →  Yes/No
Question →  [CLIP Text Encoder]   →  text_feat  (512d)
```

**Why CLIP?** CLIP was trained on 400M image-text pairs to produce semantically aligned embeddings for both modalities in the same 512d space. This means image and text vectors already "speak the same language" before fusion.

**Why freeze the encoders?** CLIP already contains rich, general-purpose knowledge. Fine-tuning it on 80K samples risks catastrophic forgetting. By freezing, we train only the small fusion head — fast, stable, and reproducible.

**Embedding caching:** Since encoders are frozen, the same image always produces the same embedding. We precompute and cache all embeddings once (`scripts/precompute_embeddings.py`), so training only touches small vectors. No image loading during training.

---

## The Four Fusion Strategies

This is the core of the project. Each strategy represents a different philosophy for combining image and text vectors.

### 1. ConcatMLP (Baseline)
Concatenate image and text vectors end-to-end, pass through an MLP.

```
image_feat (512) + text_feat (512)  →  concat  →  (1024)  →  MLP  →  Yes/No
```

Simple but strong. The MLP sees all 1024 dimensions and can learn arbitrary mappings. No explicit cross-modal interaction.

### 2. BilinearFusion
Project both vectors into the same hidden space, then combine via element-wise multiplication.

```
image_feat  →  Linear  →  img (1024)
text_feat   →  Linear  →  txt (1024)
                img * txt  →  MLP  →  Yes/No
```

Element-wise multiplication explicitly captures interactions between corresponding dimensions. If dimension 42 encodes "dog-ness" in both image and text, their product is high — directly signaling relevance. Concatenation misses this.

### 3. CrossAttentionFusion
Let the text vector attend over image features using transformer-style cross-attention.

```
query = text_feat,  key = image_patches (196×512),  value = image_patches
attended = MultiheadAttention(query, key, value)
out = LayerNorm(attended + text_feat)  →  Classifier  →  Yes/No
```

Uses patch-level image features (196 patches, each 512d) so the question can focus on specific image regions. More expressive than global features.

### 4. CrossAttentionFusionV2
Multi-layer cross-attention with feedforward networks between layers — a proper transformer decoder block.

```
text_feat (query)
    ↓
[Cross Attention]  →  attend over 196 image patches
    ↓
[Feed Forward Network]  →  non-linear transformation
    ↓
[Cross Attention]  →  refine again
    ↓
[Feed Forward Network]
    ↓
Classifier  →  Yes/No
```

Multiple attention layers allow iterative refinement — first pass finds relevant regions, second pass focuses within them.

---

## LoRA Experiment

On top of the four fusion strategies, we run a **LoRA (Low-Rank Adaptation)** experiment — adapting CLIP's visual encoder to VQA without full fine-tuning.

**Why LoRA and not full fine-tuning?**
Fine-tuning all 150M CLIP params on 80K samples causes catastrophic forgetting and severe overfitting. LoRA instead injects tiny trainable matrices alongside frozen weights:

```
Original:    output = x @ W.T          (W frozen, 512×512 = 262K params)
With LoRA:   output = x @ W.T  +  x @ A.T @ B.T × (α/r)
             A: (rank×512), B: (512×rank) — only 8,192 params per layer
```

B is initialized to zero so LoRA starts as identity — training begins from a stable point.

We inject LoRA into the last 4 transformer blocks of CLIP's visual encoder (blocks 8-11 out of 12) — only task-specific layers adapt, early universal features stay frozen.

```
Trainable params: 294,912 / 151,572,225 (0.19%)
```

Comparison: **bilinear + frozen CLIP** vs **bilinear + LoRA CLIP**

---

## Results

### Frozen Encoder Comparison

| Model | Val Acc | F1 | Params | Stopped at |
|---|---|---|---|---|
| BilinearFusion | **61.0%** | **0.628** | 657K | Epoch 13 |
| ConcatMLP | 58.6% | 0.577 | 657K | Epoch 15 |
| CrossAttentionFusion | 57.2% | 0.587 | 1.18M | Epoch 13 |
| CrossAttentionFusionV2 | 56.5% | 0.612 | 6.43M | Epoch 14 |

### Key Findings

**1. Bilinear fusion wins convincingly** — same parameters as ConcatMLP, higher accuracy AND F1. Explicit cross-modal interactions via element-wise multiplication are genuinely better than simple concatenation.

**2. More layers hurt CrossAttention** — V2 has 10x more parameters than Bilinear but the lowest accuracy. The extra depth doesn't help when the text query is a single global vector. Attention needs a rich sequence to attend over.

**3. CrossAttention is the most balanced predictor** — its F1 (0.587) is higher than ConcatMLP's F1 (0.577) despite lower accuracy, meaning it makes more balanced yes/no predictions rather than over-predicting one class.

**4. Overfitting was a real challenge** — initial runs showed train accuracy 94% vs val accuracy 61% (33% gap). Fixed with increased dropout (0.3→0.5), stronger weight decay (1e-4→1e-3), reduced hidden dim (1024→512), and early stopping (patience=5).

---

## Project Structure

```
glimpse/
  src/
    models.py            ← all four fusion models
    dataset.py           ← VQADataset (loads cached embeddings)
    dataset_finetune.py  ← VQAFinetuneDataset (loads raw images for LoRA)
    train.py             ← shared training loop with early stopping
    train_lora.py        ← LoRA training loop (runs CLIP live)
    evaluate.py          ← metrics, comparison table, plots
    lora.py              ← LoRA implementation + CLIP injection
    utils.py             ← device, seeding, logging
  scripts/
    precompute_embeddings.py  ← run once to cache CLIP features
    run_training.py           ← train all fusion models
    run_lora_training.py      ← train bilinear + LoRA CLIP
  configs/
    default.yaml         ← all hyperparameters
  data/
    VQA_annotations/     ← VQA annotation JSONs
    VQA_input_questions/ ← VQA question JSONs
    VQA_input_images/    ← COCO val2014 images
    embeddings/          ← cached CLIP features (generated, not in git)
  results/
    loss_curves.png
    confusion_matrices.png
    summary.json
    *_history.json       ← per-model training history
  requirements.txt
  setup.py
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/glimpse.git
cd glimpse

# 2. Create conda environment
conda create -n glimpse python=3.11 -y
conda activate glimpse

# 3. Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git

# 4. Install project as package
pip install -e .
```

---

## Data

Download from [visualqa.org/download.html](https://visualqa.org/download.html) (Balanced Real Images section):

```
data/
  VQA_annotations/
    v2_mscoco_val2014_annotations.json
  VQA_input_questions/
    v2_OpenEnded_mscoco_val2014_questions.json
  VQA_input_images/
    val2014/   ← COCO val2014 images (~6GB)
               download: http://images.cocodataset.org/zips/val2014.zip
```

---

## Running the Project

```bash
# Step 1: Precompute CLIP embeddings (run once, takes ~3 hours on CPU)
PYTHONPATH=/path/to/glimpse python scripts/precompute_embeddings.py

# Step 2: Train all four fusion models
PYTHONPATH=/path/to/glimpse python scripts/run_training.py

# Step 3: Train bilinear + LoRA CLIP
PYTHONPATH=/path/to/glimpse python scripts/run_lora_training.py

# Step 4: Evaluate and compare all models
PYTHONPATH=/path/to/glimpse python src/evaluate.py
```

---

## Hyperparameters

All hyperparameters live in `configs/default.yaml` — nothing is hardcoded.

| Parameter | Value | Reason |
|---|---|---|
| embed_dim | 512 | Fixed — CLIP ViT-B/32 output size |
| hidden_dim | 512 | Reduced to limit overfitting |
| dropout | 0.5 | Strong regularization |
| batch_size | 256 | Large OK with cached embeddings |
| lr | 3e-4 | Standard AdamW starting point |
| weight_decay | 1e-3 | Strong L2 penalty |
| epochs | 50 | Early stopping decides actual cutoff |
| early_stopping_patience | 5 | Stop if val loss doesn't improve for 5 epochs |
| encoder_lr | 1e-6 | Tiny LR for LoRA params — gentle adaptation |
| LoRA rank | 8 | 295K trainable params (0.19% of CLIP) |
| LoRA alpha | 16 | Scale factor = alpha/rank = 2.0 |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| PyTorch | Model building and training |
| OpenAI CLIP (ViT-B/32) | Image and text encoding |
| HuggingFace datasets | VQA-v2 data access |
| scikit-learn | Evaluation metrics |
| matplotlib | Loss curves and confusion matrices |
| PyYAML | Config management |
| MPS (Apple Silicon M4) | Hardware acceleration |

---

## Team

| Person | Owns |
|---|---|
| Lakshay | Model architecture — all fusion strategies, LoRA, ablations |
| Teammate 2 | Data pipeline — open-ended VQA, embeddings, dataset analysis |
| Teammate 3 | Evaluation + demo — metrics, visualizations, web interface |

---

*Built step by step with full understanding. No black boxes.*
