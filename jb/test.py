import os
import numpy as np
from preprocessing import preprocess_corpus, get_word_probability_dict


def positive_sampling(corpus, window):
    idx = np.random.choice(len(corpus), size=1)[0]
    line = corpus[idx]
    rnd_ixs = np.random.choice(range(window//2+1, len(line)-window//2-1), size=10, replace=False)
    pairs = []
    step = window//2
    for rnd_ix in rnd_ixs:
        word = line[rnd_ix]
        for i in range(step):
            context_l = line[rnd_ix + i]
            context_r = line[rnd_ix + i]
            if word != context_l and (word, context_l) not in pairs:
                pairs.append((word, context_l))
            if word != context_r and (word, context_r) not in pairs:
                pairs.append((word, context_r))
    return pairs


def negative_sampling(freq_dict, positive_pairs):
    pairs = []

    def get_negative_pair():
        return [
            np.random.choice(list(freq_dict.keys()), p=list(freq_dict.values()))
            for _ in range(2)]

    pos_length = len(positive_pairs)
    neg_length = 0
    while neg_length < pos_length:
        neg_pair = get_negative_pair()
        if neg_pair not in positive_pairs and neg_pair[0] != neg_pair[1]:
            neg_length += 1
            pairs.append(neg_pair)
    return pairs


if __name__ == "__main__":
    root_folder = "./../data/"
    raw_corpus_fpath = os.path.join(root_folder, "raw_dataset.txt")
    corpus = preprocess_corpus(raw_corpus_fpath, limit=None)
    freq_dictionary = get_word_probability_dict(corpus)

    pos_samples = positive_sampling(corpus, window=5)
    print(pos_samples)

    neg_samples = negative_sampling(freq_dictionary, pos_samples)
    print(neg_samples)

