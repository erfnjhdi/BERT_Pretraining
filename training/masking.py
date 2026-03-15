import torch


def mask_tokens(input_ids, tokenizer, mask_prob=0.15):
    device = input_ids.device
    labels = input_ids.clone()

    mask = torch.rand(input_ids.shape, device=device) < mask_prob

    if mask.sum() == 0:
        i = torch.randint(0, input_ids.numel(), (1,), device=device)
        mask.view(-1)[i] = True

    labels[mask == 0] = -100

    masked_input = input_ids.clone()
    vocab_size = len(tokenizer.vocab_list)
    mask_token_id = tokenizer.token_to_id["[MASK]"]

    rand = torch.rand(input_ids.shape, device=device)

    # 80% -> [MASK]
    rand_mask = mask & (rand < 0.8)
    masked_input[rand_mask] = mask_token_id

    # 10% -> random token
    rand_mask = mask & (rand >= 0.8) & (rand < 0.9)
    masked_input[rand_mask] = torch.randint(0, vocab_size, (rand_mask.sum(),), device=device)

    # 10% -> keep unchanged

    return masked_input, labels
