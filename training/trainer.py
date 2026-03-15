import torch
from torch.utils.data import DataLoader, Subset

from training.masking import mask_tokens


def run_epoch(model, dataloader, tokenizer, optimizer, criterion, device="cpu"):
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch.to(device)
        masked_input, labels = mask_tokens(input_ids, tokenizer)

        logits = model(masked_input)
        B, T, V = logits.size()
        loss = criterion(logits.view(-1, V), labels.view(-1))

        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(model, train_loader, valid_loader, test_loader,
                tokenizer, criterion, optimizer, num_epochs=15, device="cpu"):
    train_losses = []
    valid_losses = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = run_epoch(model, train_loader, tokenizer, optimizer, criterion, device=device)

        model.eval()
        with torch.no_grad():
            valid_loss = run_epoch(model, valid_loader, tokenizer, optimizer, criterion, device=device)

        print(f"Epoch {epoch:02d} | Train loss: {train_loss:.4f} | Valid loss: {valid_loss:.4f}")

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

    model.eval()
    with torch.no_grad():
        test_loss = run_epoch(model, test_loader, tokenizer, optimizer, criterion, device=device)

    print(f"\nTest loss: {test_loss:.4f}")

    return train_losses, valid_losses, test_loss


def make_overfit_loader(dataloader, num_samples=4):
    dataset = dataloader.dataset
    indices = list(range(num_samples))
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=num_samples,
        collate_fn=dataloader.collate_fn,
        shuffle=False,
    )
