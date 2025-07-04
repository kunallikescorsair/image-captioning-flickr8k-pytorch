# Image Captioning on Flickr8k using ResNet + Attention + LSTM

This project focuses on building an image captioning model using the Flickr8k dataset. It combines a pre-trained ResNet-50 CNN for feature extraction with an attention-based LSTM decoder to generate descriptive captions for images. The implementation is done using PyTorch and follows an end-to-end pipeline, including data preprocessing, model training, evaluation, and caption generation.

---

## Dataset

- **Source**: Flickr8k dataset (available on Kaggle)
- **Content**: 8,000 images, each paired with five human-written captions
- **Split**:
  - 6,000 images for training
  - 1,000 images for validation
  - 1,000 images for testing

Captions were cleaned, tokenized, and encoded. A vocabulary was constructed by filtering out rare words (occurrence < 5). Special tokens like `<start>`, `<end>`, `<pad>`, and `<unk>` were added for training.

---

## Models

Two models were implemented for comparison:

### 1. VGG16 + LSTM (Baseline)
- Uses pre-extracted 4096-dimensional VGG16 features
- Single-layer LSTM decoder
- Decoding: Greedy and Beam Search

### 2. ResNet50 + Attention + LSTM (Final)
- Extracts 49×2048 spatial features from ResNet-50
- Bahdanau-style attention mechanism
- LSTMCell-based decoder for token-level generation
- Trained directly on raw images

---

## Evaluation

Models were evaluated using BLEU scores (BLEU-1 to BLEU-4) on the test set (1000 images). Beam search was set with a width of 3.

### BLEU Score Comparison:

| Model                     | Decoding      | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|--------------------------|---------------|--------|--------|--------|--------|
| VGG16 + LSTM             | Greedy        | 0.5484 | 0.3588 | 0.2311 | 0.1474 |
| VGG16 + LSTM             | Beam Search   | 0.5721 | 0.3839 | 0.2538 | 0.1649 |
| ResNet + Attention + LSTM| Greedy        | 0.5720 | 0.3927 | 0.2636 | 0.1713 |
| ResNet + Attention + LSTM| Beam Search   | **0.6146** | **0.4311** | **0.2954** | **0.1933** |

---

## Example Output

**Ground Truth Captions:**
- a dirt biker flies through the air  
- a man in red jumps his motocross bike  
- someone on a bike is moving in midair  

**Predicted Caption (Beam Search):**
- a dirt biker is riding a dirt bike

While not as dynamic as the ground truth, the predicted caption is semantically accurate and fluent.

---

## File Overview

- `Image_Captioning_Kunal.ipynb` – Final notebook with all code and outputs
- `README.md` – Project documentation

---

## Running Instructions

1. Download the Flickr8k dataset and structure it as required
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
