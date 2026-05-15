# Glimpse

Figuring out the best way to combine image and text features for yes/no visual questions.

I took CLIP, a model that already knows how to match images with text, and tried 8 different ways of plugging its outputs into a classifier to answer questions like "Is there a dog in this photo?" The core question was: does the fusion method matter once the encoder is frozen? Turns out simpler is almost always better, which is the more interesting result.


## The task

The two experiment tracks actually solve slightly different versions of the problem, which is worth being upfront about.

The CLIP fusion track is binary classification: given an image and a yes/no question, predict yes or no. I used VQA-v2 val2014 and filtered it down to yes/no questions only — 80,537 questions total, split 80/20 into train and val. Random chance here is 50%.

The ViT+BERT track is the full VQA task: given an image and any open-ended question, predict the answer from a vocabulary of 3129 possible answers (the most frequent answers in the dataset, covering ~90% of VQA v2). It trains on train2014 (~443k questions) and evaluates on val2014. The metric is the official VQA soft score — not plain accuracy — because each question was answered by 10 human annotators and answers aren't always unanimous.


## Two experiment tracks

### Track 1: CLIP frozen, different fusion heads

CLIP (ViT-B/32) runs once on all images and questions, and I save the embeddings to disk. Then I train only a small fusion head on top of those fixed features. This is fast to iterate — training each model takes minutes, not hours. This track uses val2014 only (~80,537 yes/no questions), split 80/20 into train and val.

Each image gives a global 512-d vector (CLS token from layer 12), 49 patch tokens at 512-d each (spatial features from layer 12), and patch tokens from layers 4 and 8 for models that use multi-layer features. Each question gives a pooled 512-d vector and per-token embeddings for all 77 positions.

I tried 8 different ways of combining these into a yes/no answer:

| Model | What it does | Val Acc | F1 | Params |
|---|---|---|---|---|
| **GeometryFusion** | Decomposes CLIP's alignment into 3 signals: cosine similarity, residual (img minus txt), and interaction (img times txt). No learned projection, just MLP on top | **63.3%** | 0.662 | 657K |
| BilinearFusion | Projects both vectors to a shared space, multiplies element-wise, then MLP | 60.2% | 0.600 | 657K |
| ConcatMLP | Concatenates image and text vectors, feeds into a 3-layer MLP | 58.6% | 0.594 | 657K |
| LayerAdaptiveFusion | Pulls patch features from CLIP layers 4, 8, and 12. A small gating network learns which layer matters most per question | 57.2% | 0.585 | 6.5M |
| CrossAttentionFusion | Text vector queries the 49 image patch tokens via cross-attention | 57.1% | 0.608 | 1.2M |
| MultiGlimpse | Cross-attention at 3 spatial scales (2x2 to 4x4 to 7x7), same shared weights at each scale | 56.9% | 0.610 | 3.3M |
| TokenGrounding | Each word in the question attends to image patches independently, then pooled | 56.3% | 0.563 | 6.4M |
| CrossAttentionFusionV2 | Stacked cross-attention blocks (transformer decoder style), 2 layers deep | 55.9% | 0.639 | 6.4M |

I also ran a LoRA experiment, injecting trainable low-rank matrices into the last 4 blocks of CLIP's visual encoder (rank=8, ~295K trainable params out of CLIP's 150M total), to see if letting the encoder adapt slightly would help. The checkpoint is at `results/checkpoints/lora_bilinear_best.pt`.

### Track 2: ViT+BERT end-to-end

A completely separate approach with no CLIP. ViT-B/16 (pretrained on ImageNet) encodes images into 197 patch tokens at 768-d. BERT-base encodes questions into token embeddings at 768-d. A bidirectional co-attention module then fuses them — text attends to image and image attends to text, alternating for several layers, ending with a gated fusion that blends the two CLS representations into a single vector for classification.

This track is trained on the **full VQA task** — 3129 possible answers (the top answers by frequency, covering about 90% of VQA v2), not filtered to yes/no. It trains on train2014 images (~443k questions) and evaluates on val2014. The encoders are fine-tuned jointly with the fusion head, but at a much smaller learning rate (1/20th of the fusion layers) to avoid destroying pretrained features.

The accuracy metric here is the **VQA soft score**, which is the official VQA v2 evaluation metric. Since each question was answered by 10 human annotators, the score for a predicted answer is `min(how many annotators gave that answer / 3, 1.0)`. An answer that 9 out of 10 annotators agreed on scores 1.0. An answer that 2 out of 10 gave scores 0.67. This better reflects real answer ambiguity than exact-match accuracy.

This went through 4 iterations:

| Version | Architecture | Best Val Acc | What changed |
|---|---|---|---|
| v1 | MobileNetV3-small + DistilBERT, 1000-class, one-directional cross-attention | ~38% | Proof of concept on a 10k/2k subset for speed. Hard labels only. |
| v2 | ViT-B/16 + BERT, bidirectional co-attention (depth=2), soft labels, gated fusion | 57.6% | Upgraded encoders and added soft VQA labels — but a bug meant soft labels never actually loaded, so it trained on hard one-hot targets the whole time |
| v3 | Same architecture, depth=4 | 66.9% | Fixed 4 bugs: (1) question_id not returned from data loader so soft label lookup silently fell back to hard labels every sample, (2) PAD query tokens had no mask after cross-attention and leaked corrupted signal through LayerNorm into image KV, (3) text-to-image masking was missing — only image-to-text direction was masked, (4) image pooling switched from mean-pool to attention pooling |
| v4 | Same + label smoothing | **67.6%** | Was accidentally using DistilBERT's tokenizer vocabulary with a BERT model — completely wrong token IDs. Switched to BERT tokenizer, added label smoothing. |


## What I found

With frozen CLIP embeddings, simpler fusion consistently wins.

GeometryFusion takes the top spot by doing something almost embarrassingly simple: it just decomposes what CLIP already computed into three components and feeds them into an MLP. It doesn't need to learn any projections, has the same parameter count as the weakest model, and still beats everything else by a clear margin.

The cross-attention models underperform despite being architecturally more sophisticated. The reason is that CLIP's layer-12 patch tokens aren't really spatial features anymore. After 12 rounds of self-attention mixing, every "patch" already contains global context from the whole image. Attending over 49 nearly-identical summaries of the same scene doesn't give the model anything new to work with. BilinearFusion and GeometryFusion avoid this problem by working with the global CLS token directly.

More parameters made things worse across the board. When the input signal can only support around 60% accuracy, extra model capacity just means faster overfitting to noise. Every model hit early stopping between epochs 4 and 16.

The end-to-end ViT+BERT approach (67.6%) beat the frozen CLIP approach (63.3%) because the encoder could actually adapt to the task rather than being locked into image-text matching features. The gap would likely grow with more training data.


## Project structure

```
src/                    core code — models, dataset, training loop, evaluation
  models.py             all 8 fusion architectures + shared building blocks
  dataset.py            loads precomputed CLIP embeddings (per-key or single file)
  train.py              shared training loop for all CLIP fusion models
  train_lora.py         training loop for LoRA experiment
  evaluate.py           loads checkpoints, runs comparison table + plots
  lora.py               LoRA injection into CLIP visual encoder
  dataset_finetune.py   dataset for LoRA (loads raw images, not cached embeddings)
  utils.py              device detection, logger, seed

scripts/                runnable scripts
  precompute_embeddings.py   runs CLIP on all images/questions, saves to disk
  merge_chunks.py            memory-efficient merge when embeddings saved in chunks
  run_training.py            trains all 8 CLIP fusion models sequentially
  run_lora_training.py       trains LoRA + BilinearFusion
  train_mobilenet_distilbert.py   ViT+BERT track v1
  train_vit_bert_v1.py            ViT+BERT track v2
  train_vit_bert_v2.py            ViT+BERT track v3 (bug fixes)
  train_vit_bert_v3.py            ViT+BERT track v4 (final)

configs/
  default.yaml           all hyperparameters in one place

results/
  summary.json           comparison table for all 8 CLIP models
  loss_curves.png        training curves
  confusion_matrices.png
  *_history.json         per-model epoch-by-epoch metrics
  vit_bert/              training logs from the ViT+BERT experiments
```


## How to run

```bash
pip install -e .
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
```

Data goes here (not included). The CLIP fusion track only needs val2014. The ViT+BERT track also needs train2014:
```
data/VQA_input_questions/v2_OpenEnded_mscoco_val2014_questions.json
data/VQA_annotations/v2_mscoco_val2014_annotations.json
data/VQA_input_images/val2014/

# additionally for ViT+BERT:
data/VQA_input_questions/v2_OpenEnded_mscoco_train2014_questions.json
data/VQA_annotations/v2_mscoco_train2014_annotations.json
data/VQA_input_images/train2014/
```

```bash
# step 1: extract CLIP embeddings and save to disk (run once, takes 2-3 hrs on MPS/CPU)
python scripts/precompute_embeddings.py

# step 2: train all 8 fusion models
python scripts/run_training.py

# step 3: evaluate and compare
python src/evaluate.py

# smoke test to verify all model shapes and gradient flow
python test_models.py
```

For the LoRA experiment:
```bash
python scripts/run_lora_training.py
```

For ViT+BERT (needs GPU, separate data setup with train2014 split):
```bash
python scripts/train_vit_bert_v3.py
```

All hyperparameters are in `configs/default.yaml`. No hardcoded values in source files.


## Requirements

- Python 3.9+
- PyTorch (CUDA or MPS recommended; CPU works but the precompute step is slow)
- openai-clip, torchvision, tqdm, pillow, pyyaml, scikit-learn, matplotlib
- ViT+BERT scripts additionally need: timm, transformers, pandas
