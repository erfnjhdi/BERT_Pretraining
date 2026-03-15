import torch
import torch.nn as nn

import config
from data.preprocessing import load_data, split_data
from data.tokenizer import train_tokenizer
from data.dataset import make_dataloaders
from model.bert import build_model
from training.trainer import run_epoch, train_model, make_overfit_loader
from evaluation.embeddings import get_embedding, cosine_similarity_matrix
from visualization.plots import plot_loss_curves, plot_similarity_heatmap


def main():
    # Data
    raw_text = load_data(config.CORPUS_PATH)
    train_text, valid_text, test_text = split_data(
        raw_text, config.TRAIN_RATIO, config.VALID_RATIO, config.RANDOM_SEED
    )

    # Tokenizer
    tokenizer = train_tokenizer(train_text, config.VOCAB_SIZE)
    vocab_size = len(tokenizer.token_to_id)

    # DataLoaders
    train_loader, valid_loader, test_loader = make_dataloaders(
        train_text, valid_text, test_text,
        tokenizer, config.SEQ_LEN, config.BATCH_SIZE,
    )

    # Overfit sanity check
    print("\nOverfitting sanity check:\n")
    overfit_model = build_model(vocab_size, config.EMBED_DIM, config.NUM_HEADS,
                                config.HIDDEN_DIM, config.NUM_LAYERS, config.SEQ_LEN)
    overfit_model.train()
    overfit_criterion = nn.CrossEntropyLoss()
    overfit_optimizer = torch.optim.Adam(overfit_model.parameters(), lr=1e-3)
    overfit_loader = make_overfit_loader(test_loader)

    for epoch in range(1, 5001):
        loss = run_epoch(overfit_model, overfit_loader, tokenizer,
                         overfit_optimizer, overfit_criterion, device=str(config.DEVICE))
        if epoch % 500 == 0:
            print(f"  Epoch {epoch:04d} | Loss: {loss:.6f}")
        if loss < 0.1:
            print(f"  Converged at epoch {epoch} with loss {loss:.6f}")
            break

    print("Overfitting sanity check passed\n")

    # Full training
    print("\nFull training:\n")
    model = build_model(vocab_size, config.EMBED_DIM, config.NUM_HEADS,
                        config.HIDDEN_DIM, config.NUM_LAYERS, config.SEQ_LEN)
    model.to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_losses, valid_losses, test_loss = train_model(
        model, train_loader, valid_loader, test_loader,
        tokenizer, criterion, optimizer,
        num_epochs=config.NUM_EPOCHS,
        device=str(config.DEVICE),
    )

    # Loss curves
    plot_loss_curves(train_losses, valid_losses)

    # Embedding analysis
    print("\nEmbedding analysis:\n")
    all_embeddings = get_embedding(config.EMBED_PROBE_TEXT, tokenizer, model, device=str(config.DEVICE))

    emb = {label: all_embeddings[idx] for label, idx in config.EMBED_LABELS.items()}

    conf_mat, labels = cosine_similarity_matrix(emb)
    print("Cosine similarity matrix:")
    print(conf_mat)

    plot_similarity_heatmap(conf_mat, labels)


if __name__ == "__main__":
    main()
