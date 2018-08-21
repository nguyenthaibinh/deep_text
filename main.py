from tensorflow.contrib import learn
import torch
from torch import optim
import numpy as np
import data
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import os


def preprocess():
    print("Load data.")
    # X_text, Y, _, _ = data.load_data_and_labels_from_csv(dataset="yelp_review_polarity")
    # print("Y:", Y[:10])
    X_1, X_2, Y = data.load_quora_data()
    print("X_1.size:", len(X_1))
    print("X_2.size:", len(X_2))
    print("Y.size:", len(Y))

    X_merged = []
    X_merged.extend(X_1)
    X_merged.extend(X_2)

    print("X_merged.size:", len(X_merged))
    print("X_merged.type:", type(X_merged))
    print("X_merged:", X_merged[:10])
    n_len = len(X_merged)

    print("Map sentences to sequence of word id.")
    max_document_length = max([len(x.split(" ")) for x in X_merged])
    print("Max document length:", max_document_length)

    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    X_merged = np.array(list(vocab_processor.fit_transform(X_merged)))
    X1 = X_merged[:int(n_len / 2)]
    X2 = X_merged[int(n_len / 2):]
    Y = np.array(Y)

    print("X_1.size:", X1.shape)
    print("X_2.size:", X2.shape)
    print("Y.size:", Y.shape)

    # Shuffle the data
    print("Shuffle the data.")
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(Y)))
    x1_shuffled = X1[shuffle_indices]
    x2_shuffled = X2[shuffle_indices]
    y_shuffled = Y[shuffle_indices]

    # Split train/test set
    print("Split train/test set")
    dev_sample_index = -1 * int(0.1 * float(len(Y)))
    x1_train, x1_dev = x1_shuffled[:dev_sample_index], x1_shuffled[dev_sample_index:]
    x2_train, x2_dev = x2_shuffled[:dev_sample_index], x2_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del X1, X2, Y, x1_shuffled, x2_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x1_train, x2_train, y_train, vocab_processor, x1_dev, x2_dev, y_dev


def main(argv=None):
    # args = parse_args()
    res = preprocess()
    x1_train, x2_train, y_train, vocab_processor, x1_dev, x2_dev, y_dev = res

    print("num positive:", len(y_dev[y_dev == 1]))
    print("num negative:", len(y_dev[y_dev == 0]))
    print("all:", len(y_dev))

if __name__ == '__main__':
    main()
