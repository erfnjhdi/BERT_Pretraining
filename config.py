import torch

# Data
CORPUS_PATH = "synthetic_corpus.txt"
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
RANDOM_SEED = 42

# Tokenizer
VOCAB_SIZE = 100

# Dataset / DataLoader
SEQ_LEN = 8
BATCH_SIZE = 16

# Model
EMBED_DIM = 32
NUM_HEADS = 2
HIDDEN_DIM = 64
NUM_LAYERS = 1

# Training
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
MASK_PROB = 0.15

# Device
DEVICE = torch.device("cpu")

# Embedding analysis
EMBED_WORDS = ["king", "queen", "the village", "the"]
