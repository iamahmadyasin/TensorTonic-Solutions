import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    N = len(seqs)
    if max_len is None:
        max_len = max((len(s) for s in seqs), default=0)

    out = np.full((N, max_len), pad_value, dtype=int)
    for i, s in enumerate(seqs):
        trunc = s[:max_len]
        out[i, :len(trunc)] = trunc
    return out