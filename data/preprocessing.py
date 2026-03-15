import random


def load_data(path):
    with open(path, "r") as f:
        return f.readlines()


def split_data(lines, train_ratio=0.8, valid_ratio=0.1, seed=42):
    random.seed(seed)
    random.shuffle(lines)

    n = len(lines)
    n_train = int(train_ratio * n)
    n_valid = int(valid_ratio * n)

    train = lines[:n_train]
    valid = lines[n_train:n_train + n_valid]
    test = lines[n_train + n_valid:]

    return train, valid, test
