import os
import numpy as np
import matplotlib.pyplot as plt

PLOTS_DIR = os.path.join(os.path.dirname(__file__))


def plot_loss_curves(train_losses, valid_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, marker="o", label="Train loss")
    plt.plot(epochs, valid_losses, marker="s", label="Valid loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(list(epochs)[::5])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "loss_curves.png"))
    plt.show()


def plot_similarity_heatmap(conf_mat, labels):
    vmin = conf_mat.min()
    vmax = conf_mat.max()

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(conf_mat, cmap="viridis", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{conf_mat[i, j]:.2f}",
                    ha="center", va="center", color="white")

    ax.set_title("Cosine Similarity Between Embeddings")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "similarity_heatmap.png"))
    plt.show()
