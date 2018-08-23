from tensorflow.contrib import learn
import torch
from torch import nn
from torch import optim
import numpy as np
import data
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import os
from encoders.cnn_encoder import DualCNN
from encoders.mean_encoder import DualMean

GLOVE_FILE = "./data/word_vectors/glove.840B.300d.txt"
W2V_FILE = "./data/word_vectors/GoogleNews-vectors-negative300.bin"


def preprocess(top=0):
    print("Load data.")
    # X_text, Y, _, _ = data.load_data_and_labels_from_csv(dataset="yelp_review_polarity")
    # print("Y:", Y[:10])
    X_1, X_2, Y = data.load_quora_data(top=top)
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


def train(model, x1_train, x2_train, y_train, vocab_processor,
          x1_dev, x2_dev, y_dev, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.BCELoss(weight=None, size_average=False)
    batch_size = int(args.batch_size)

    model.train()
    for epoch in range(args.epochs):
        running_losses = []

        tmp_loss = 0

        # Generate batch
        batches = data.batch_iter(list(zip(x1_train, x2_train, y_train)),
                                  batch_size)
        for batch in batches:
            x1_batch, x2_batch, y_batch = zip(*batch)
            x1_batch = Variable(torch.LongTensor(x1_batch))
            x2_batch = Variable(torch.LongTensor(x2_batch))
            y_batch = Variable(torch.FloatTensor(y_batch))

            """
            print("x1_batch.size:", x1_batch.size())
            print("x2_batch.size:", x2_batch.size())
            print("y_batch.size:", y_batch.size())
            """

            if torch.cuda.is_available():
                x1_batch = x1_batch.cuda()
                x2_batch = x2_batch.cuda()
                y_batch = y_batch.cuda()

            preds, classes = model(x1_batch, x2_batch)

            # print("preds.size:", preds.size())

            # Gradient descent
            optimizer.zero_grad()
            loss = loss_func(preds, y_batch)
            loss.backward()
            optimizer.step()

            tmp_loss += loss.data[0].item()
            running_losses.append(loss.data[0].item())

            """
            y_truth = y_batch.byte()

            print("classes:", classes.data)
            print("y_truth:", y_truth.data)
            print("y_batch:", y_batch.data)

            print("type(classes):", type(classes))
            print("type(y_batch):", type(y_batch))
            print("type(y_truth):", type(y_truth))
            """

        epoch_loss = sum(running_losses) / len(running_losses)

        train_loss, train_acc = eval(model, x1_train, x2_train, y_train, batch_size)
        dev_loss, dev_acc = eval(model, x1_dev, x2_dev, y_dev, batch_size)

        print("Epoch: {}, loss: {}, train_acc: {}, dev_acc: {}".format(epoch + 1,
                                                                       epoch_loss,
                                                                       train_acc,
                                                                       dev_acc))
        if (epoch + 1) % args.checkpoint_interval == 0:
            model_name = model.__class__.__name__
            save(model, save_dir=args.save_dir,
                 save_prefix="snapshot", model_name=model_name,
                 steps=epoch + 1)
    return 0


def eval(model, x1_dev, x2_dev, y_dev, batch_size, verbose=False):
    model.eval()
    loss_func = nn.BCELoss(weight=None, size_average=False)

    corrects = 0.0
    running_losses = []

    batches = data.batch_iter(list(zip(x1_dev, x2_dev, y_dev)), batch_size)
    for batch in batches:
        x1_batch, x2_batch, y_batch = zip(*batch)
        x1_batch = Variable(torch.LongTensor(x1_batch))
        x2_batch = Variable(torch.LongTensor(x2_batch))
        y_batch = Variable(torch.FloatTensor(y_batch))

        if torch.cuda.is_available():
            x1_batch = x1_batch.cuda()
            x2_batch = x2_batch.cuda()
            y_batch = y_batch.cuda()

        preds, classes = model(x1_batch, x2_batch)
        loss = loss_func(preds, y_batch)
        running_losses.append(loss.data[0].item())

        y_truth = y_batch.byte()

        tmp_corrects = (classes.data == y_truth.data).sum()
        corrects += tmp_corrects

    size = len(x1_dev)
    avg_loss = sum(running_losses) / len(running_losses)
    accuracy = (100.0 * corrects / size)

    return avg_loss, accuracy


def save(model, save_dir, save_prefix, model_name, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_{}_steps_{}.pt'.format(save_prefix, model_name, steps)
    torch.save(model.state_dict(), save_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Dual Text Encoders')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 1000)')
    parser.add_argument('--top', type=int, default=0, metavar='N',
                        help='Get top rows of the quora data (default: 0)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--checkpoint-interval', type=int, default=50, metavar='N',
                        help='number of step to save the model (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--embed-size', type=int, default=128, metavar='N',
                        help='the dimensionality of the embedding space (default: 128)')
    parser.add_argument('--num-filters', type=int, default=128, metavar='N',
                        help='the number of filters (default: 128)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--model', type=str, default="cnn",
                        help='the embedding model')
    parser.add_argument('--save-dir', type=str, default="./checkpoints/",
                        help='the embedding model')
    parser.add_argument('--word-vectors', type=str, default="none",
                        help='the pre-trained word vectors')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def main(argv=None):
    args = parse_args()
    res = preprocess(top=args.top)
    x1_train, x2_train, y_train, vocab_processor, x1_dev, x2_dev, y_dev = res

    print("num positive:", len(y_dev[y_dev == 1]))
    print("num negative:", len(y_dev[y_dev == 0]))
    print("all:", len(y_dev))

    if args.word_vectors == "glove":
        word_vectors = data._load_glove(vocab_processor, GLOVE_FILE)
    elif args.word_vectors == "word2vec":
        word_vectors = data._load_word2vec(vocab_processor, W2V_FILE)
    else:
        word_vectors = None

    if args.model == "cnn":
        model = DualCNN(sequence_length=x1_train.shape[1],
                        num_classes=2,
                        vocab_size=len(vocab_processor.vocabulary_),
                        embed_size=args.embed_size,
                        word_vectors=word_vectors,
                        filter_sizes=[3, 4, 5],
                        num_filters=args.num_filters,
                        l2_reg=0.01)
    elif args.model == "mean":
        model = DualMean(sequence_length=x1_train.shape[1], num_classes=2,
                         vocab_size=len(vocab_processor.vocabulary_),
                         embed_size=args.embed_size,
                         word_vectors=word_vectors,
                         l2_reg=0.01)
    else:
        print("Wrong model!")
        return 0

    if torch.cuda.is_available():
        model = model.cuda()

    print("train.size:", x1_train.shape[0])
    print("test.size:", x1_dev.shape[0])

    train(model, x1_train, x2_train, y_train, vocab_processor,
          x1_dev, x2_dev, y_dev, args)

    """
    print("EVALUATION!")
    print("=======================")
    model.print_parameters()
    loss = eval(model, x1_dev, x2_dev, y_dev, batch_size=args.batch_size)
    print("Evaluation: loss: {}".format(loss))
    """

if __name__ == '__main__':
    main()
