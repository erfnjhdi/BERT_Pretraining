import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=32):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        corpus = "".join(text).replace("\n", " [SEP] ")
        self.ids = self.tokenizer.encode_ids(corpus)

        print(f"Number of tokens in the dataset: {len(self.ids)}")

    def __len__(self):
        return max(0, len(self.ids) - (self.seq_len - 1))

    def __getitem__(self, idx):
        x = self.ids[idx: idx + self.seq_len]
        return torch.tensor(x, dtype=torch.long)


def make_dataloaders(train_text, valid_text, test_text, tokenizer, seq_len, batch_size):
    train_dataset = TextDataset(train_text, tokenizer, seq_len)
    valid_dataset = TextDataset(valid_text, tokenizer, seq_len)
    test_dataset = TextDataset(test_text, tokenizer, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
