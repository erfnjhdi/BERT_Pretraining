from collections import Counter


class CharBPETokenizer:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.vocab = set()
        self.special_tokens = ["[UNK]", "[MASK]", "[SEP]"]

    def train(self, text):
        tokens = list(text)
        vocab = Counter(tokens)

        while len(vocab) < self.vocab_size:
            pair_counts = Counter()

            for j in range(len(tokens) - 1):
                pairs = (tokens[j], tokens[j + 1])
                pair_counts[pairs] += 1

            if not pair_counts:
                break

            best_pair, _ = pair_counts.most_common(1)[0]

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens
            vocab = Counter(tokens)

        vocab_set = set(tokens) | set(text) | set(self.special_tokens)
        self.vocab_list = sorted(list(vocab_set), key=len, reverse=True)
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab_list)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab_list)}

    def tokenize(self, text):
        tokens = []
        i = 0
        while i < len(text):
            match_found = False
            for token in self.vocab_list:
                if text.startswith(token, i):
                    tokens.append(token)
                    i += len(token)
                    match_found = True
                    break
            if not match_found:
                tokens.append("[UNK]")
                print(f"[WARNING] Character '{text[i]}' not in vocabulary, replaced with [UNK]")
                i += 1
        return tokens

    def encode_ids(self, text):
        tokens = self.tokenize(text)
        return [self.token_to_id[t] for t in tokens]

    def detokenize(self, tokens):
        return "".join(tokens)

    def decode_ids(self, ids):
        tokens = [self.id_to_token[id] for id in ids]
        return self.detokenize(tokens)


def train_tokenizer(train_text, vocab_size=100):
    tokenizer = CharBPETokenizer(vocab_size=vocab_size)
    corpus = "".join(train_text).replace("\n", " ")
    tokenizer.train(corpus)
    return tokenizer
