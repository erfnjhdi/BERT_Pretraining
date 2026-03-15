import torch
import torch.nn.functional as F


def get_embedding(word, tokenizer, model, device="cpu"):
    model.eval()
    model.to(device)

    with torch.no_grad():
        ids = tokenizer.encode_ids(word)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        encoded = model.encode(input_ids)

    return encoded.squeeze(0)


def cosine_similarity_matrix(embeddings_dict):
    labels = list(embeddings_dict.keys())
    n = len(labels)
    conf_mat = torch.zeros(n, n)

    for i, w1 in enumerate(labels):
        for j, w2 in enumerate(labels):
            conf_mat[i, j] = F.cosine_similarity(
                embeddings_dict[w1], embeddings_dict[w2], dim=0
            )

    return conf_mat, labels
