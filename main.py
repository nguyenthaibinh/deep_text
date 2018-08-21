from tensorflow.contrib import learn
import torch
from torch import optim
import numpy as np
import data
from models.cnntext import TextCNN, TextCNN1D
from models.textnet import TextNet
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import os


def preprocess():
    print("Load data.")
    # X_text, Y, _, _ = data.load_data_and_labels_from_csv(dataset="yelp_review_polarity")
    # print("Y:", Y[:10])
    X_1, X_2, Y = data.load_quora_data()
    X_merged = X_1.append(X_2)
    n_len = len(X_merged)

    print("Map sentences to sequence of word id.")
    max_document_length = max([len(x.split(" ")) for x in X_merged])
    print("Max document length:", max_document_length)

    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    X_merged = np.array(list(vocab_processor.fit_transform(X_merged)))
    X1 = X_merged[:int(n_len / 2)]
    X2 = X_merged[int(n_len / 2):]
    Y = np.array(Y)

    print("X_1.size:", X1.size())
    print("X_2.size:", X2.size())
    print("Y.size:", Y.size())

    # Shuffle the data
    print("Shuffle the data.")
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(Y)))
    x1_shuffled = X_1[shuffle_indices]
    x2_shuffled = X_1[shuffle_indices]
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
