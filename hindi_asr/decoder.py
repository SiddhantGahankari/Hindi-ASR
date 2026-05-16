import json

VOCAB_PATH = "data/vocab.json"

def load_vocab(path):
    with open(path) as f:
        vocab_dict = json.load(f)
    vocab_dict_rev = {v: k for k, v in vocab_dict.items()}
    return vocab_dict, vocab_dict_rev

def greedy_decode(log_probs, input_lengths, vocab_dict_rev, blank_id=0):
    predictions = log_probs.argmax(dim=-1)
    decoded = []
    for i, length in enumerate(input_lengths):
        tokens = predictions[i, :length].tolist()
        collapsed = []
        prev = None
        for t in tokens:
            if t != blank_id and t != prev:
                collapsed.append(t)
            prev = t
        text = "".join([vocab_dict_rev.get(t, "<unk>") for t in collapsed])
        decoded.append(text)
    return decoded

def compute_wer(hypotheses, references):
    total_words = 0
    total_errors = 0
    for hyp, ref in zip(hypotheses, references):
        hyp_words = hyp.strip().split()
        ref_words = ref.strip().split()
        total_words += len(ref_words)
        total_errors += edit_distance(hyp_words, ref_words)
    if total_words == 0:
        return 0.0
    return total_errors / total_words

def compute_cer(hypotheses, references):
    total_chars = 0
    total_errors = 0
    for hyp, ref in zip(hypotheses, references):
        # We pass the strings directly to edit_distance to compare characters
        total_chars += len(ref)
        total_errors += edit_distance(hyp, ref)
    if total_chars == 0:
        return 0.0
    return total_errors / total_chars

def edit_distance(hyp, ref):
    m, n = len(hyp), len(ref)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]