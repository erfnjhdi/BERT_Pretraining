# MiniBERT Masked Language Model from Scratch

A PyTorch implementation of a BERT-style masked language model built from the ground up. The project covers the full deep learning pipeline: data preprocessing, a custom BPE tokenizer, a Transformer encoder model, MLM pre-training, and embedding analysis.

This project implements the core pre-training behind foundation models like BERT. A custom Byte-Pair Encoding (BPE) tokenizer is trained on a synthetic text corpus, and a lightweight Transformer encoder is pre-trained using Masked Language Modeling (MLM). This is the same self-supervised objective used to pre-train large language models.

After training, the model's learned representations are analyzed via cosine similarity to examine how well the encoder captures semantic relationships between tokens.

## Pipeline

```
Raw text corpus
    → BPE tokenizer training
    → Tokenized PyTorch datasets
    → MiniBERT pre-training
    → Loss curve visualization
    → Embedding extraction & cosine similarity analysis
```
## Results

Train loss: 1.292, Valid loss: 1.1584 after 15 epochs

Test loss: 1.1755

## Tech Stack

### Libraries

* PyTorch
* matplotlib
* numpy


## Model Architecture

- **Tokenizer**: Character-level BPE, vocab size 100
- **Embeddings**: Learned token + positional embeddings
- **Encoder**: 1-layer Transformer encoder, 2 attention heads, hidden dim 64, embed dim 32
- **Objective**: Masked Language Modeling, 15% of tokens masked (80% `[MASK]`, 10% random, 10% unchanged)
- **Optimizer**: Adam
