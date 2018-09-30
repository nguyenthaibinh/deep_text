import os
import sys
pardir = os.path.dirname(os.getcwd())
print("pardir:", pardir)
sys.path.append(pardir)

from tensorflow.contrib import learn
import torch
from torch import nn
from torch import optim
import numpy as np
from data import data
from torch.autograd import Variable
import argparse
from encoders.cnn_encoder import CNNEncoder
from encoders.mean_encoder import MeanEncoder
from encoders.max_encoder import MaxEncoder
from encoders.lstm_encoder import LSTMEncoder
from models.dual_encoders import DualEncoders
from utils import write_vocab, _write_elapsed_time
import time
from sacred import Experiment

from tensorboard_logger import configure, log_value

QUORA_DIR = "../../datasets/pre/quora/"
DATA_FILE = os.path.join(QUORA_DIR, "quora_duplicate_questions_preprocessed.tsv")
VOCAB_FILE_1 = os.path.join(QUORA_DIR, "vocab1.csv")
VOCAB_FILE_2 = os.path.join(QUORA_DIR, "vocab1.csv")
GLOVE_FILE = "./data/word_vectors/glove.840B.300d.txt"
W2V_FILE = "./data/word_vectors/GoogleNews-vectors-negative300.bin"
TEST_SPLIT = 0.1
VALIDATION_SPLIT = 0.1
RNG_SEED = 13371447
MAX_LEN = 30
CUDA_VISIBLE_DEVICES = [0, 1, 2, 3, 4, 5, 6, 7]
FILTER_SIZES = [3, 5, 7]


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


def preprocess1(top=0, val_rate=0.1, test_rate=0.1):
    # X_text, Y, _, _ = data.load_data_and_labels_from_csv(dataset="yelp_review_polarity")
    # print("Y:", Y[:10])
    X_1, X_2, Y = data.load_quora_data(file_name=DATA_FILE, top=top)

    max_X1 = max([len(x.split(" ")) for x in X_1])
    max_X2 = max([len(x.split(" ")) for x in X_2])

    vocab1 = learn.preprocessing.VocabularyProcessor(MAX_LEN)
    vocab2 = learn.preprocessing.VocabularyProcessor(MAX_LEN)

    X1 = np.array(list(vocab1.fit_transform(X_1)))
    X2 = np.array(list(vocab2.fit_transform(X_2)))
    Y = np.array(Y)

    write_vocab(vocab1, VOCAB_FILE_1)
    write_vocab(vocab2, VOCAB_FILE_2)

    print("==================")
    print("Train/Test split")
    # X = np.stack((X1, X2), axis=1)
    # print("X1.shape:", X1)
    # print("X2.shape:", X2)
    shuffle_idx = np.random.permutation(np.arange(len(Y)))
    x1_all = X1[shuffle_idx]
    x2_all = X2[shuffle_idx]
    y_all = Y[shuffle_idx]

    test_sample_idx = -1 * int(test_rate * float(len(y_all)))
    x1_train, x1_test = x1_all[:test_sample_idx], x1_all[test_sample_idx:]
    x2_train, x2_test = x2_all[:test_sample_idx], x2_all[test_sample_idx:]
    y_train, y_test = y_all[:test_sample_idx], y_all[test_sample_idx:]

    val_sample_idx = -1 * int(val_rate * float(len(y_train)))
    x1_train, x1_val = x1_train[:val_sample_idx], x1_train[val_sample_idx:]
    x2_train, x2_val = x2_train[:val_sample_idx], x2_train[val_sample_idx:]
    y_train, y_val = y_train[:val_sample_idx], y_train[val_sample_idx:]

    print("Vocab 1 Size: {:d}".format(len(vocab1.vocabulary_)))
    print("Vocab 2 Size: {:d}".format(len(vocab2.vocabulary_)))
    print("Train/Val/Test split: {:d}/{:d}/{:d}".format(len(y_train), len(y_val), len(y_test)))
    return (x1_train, x2_train, y_train,
            x1_val, x2_val, y_val,
            x1_test, x2_test, y_test, vocab1, vocab2)


def train(model, x1_train, x2_train, y_train,
          x1_val, x2_val, y_val,
          x1_test, x2_test, y_test, args):
    """
    exp = Experiment(name='dual_text')
    exp.tag({'learning_rate': args.lr, 'weight_decay': args.l2,
             'batch_size': args.batch_size, 'embed_dim': args.embed_dim})
    """
    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.l2)
    """
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.8,
                          weight_decay=args.l2)
    """
    loss_func = nn.BCELoss()
    batch_size = int(args.batch_size)

    # logger = Logger("./logs/quora")
    configure("./logs/quora", flush_secs=5)

    for epoch in range(args.epochs):
        model.train()

        start_t = time.time()

        # Generate batch
        batches = data.batch_iter(list(zip(x1_train, x2_train, y_train)),
                                  batch_size)

        train_step(model, batches, optimizer, loss_func)

        loss_train, acc_train = eval(model, x1_train, x2_train, y_train, batch_size)
        loss_val, acc_val = eval(model, x1_val, x2_val, y_val, batch_size)
        loss_test, acc_test = eval(model, x1_test, x2_test, y_test, batch_size)

        tmp_str = "Epoch: {:3d}, loss_train: {:.4f}, acc_train: {:.4f}, "
        tmp_str += "loss_val: {:.4f}, acc_val: {:.4f}, "
        tmp_str += "loss_test: {:.4f}, acc_test: {:.4f}"

        result_str = tmp_str.format(epoch + 1, loss_train, acc_train,
                                    loss_val, acc_val, loss_test, acc_test)

        _write_elapsed_time(start_t=start_t, msg=result_str)

        # log scalar values
        log_info = {'loss_train': loss_train, 'acc_train': acc_train,
                    'loss_dev': loss_val, 'acc_dev': acc_val,
                    'loss_test': loss_test, 'acc_test': acc_test}
        """
        exp.log({'loss_train': loss_train, 'acc_train': acc_train,
                 'loss_dev': loss_val, 'acc_dev': acc_val,
                 'loss_test': loss_test, 'acc_test': acc_test})
        """

        for tag, value in log_info.items():
            # logger.scalar_summary(tag, value, epoch + 1)
            log_value(tag, value, epoch + 1)

        if (epoch + 1) % args.checkpoint_interval == 0:
            model_name = model.__class__.__name__
            save(model, save_dir=args.save_dir,
                 save_prefix="snapshot", model_name=model_name,
                 steps=epoch + 1)
    return 0


def train_step(model, batches, optimizer, loss_func):
    for batch in batches:
        x1_batch, x2_batch, y_batch = zip(*batch)
        x1_batch = Variable(torch.LongTensor(x1_batch))
        x2_batch = Variable(torch.LongTensor(x2_batch))
        y_batch = Variable(torch.FloatTensor(y_batch))

        if torch.cuda.is_available():
            x1_batch = x1_batch.cuda()
            x2_batch = x2_batch.cuda()
            y_batch = y_batch.cuda()

        preds, pred_labels = model(x1_batch, x2_batch)

        # print("preds.size:", preds.size())

        # Gradient descent
        optimizer.zero_grad()
        loss = loss_func(preds, y_batch)
        loss.backward()
        optimizer.step()


def eval(model, x1, x2, y, batch_size, verbose=False):
    model.eval()
    loss_func = nn.BCELoss()

    corrects = 0.0
    running_losses = []

    batches = data.batch_iter(list(zip(x1, x2, y)), batch_size)
    for batch in batches:
        x1_batch, x2_batch, y_batch = zip(*batch)
        x1_batch = Variable(torch.LongTensor(x1_batch))
        x2_batch = Variable(torch.LongTensor(x2_batch))
        y_batch = Variable(torch.FloatTensor(y_batch))

        if torch.cuda.is_available():
            x1_batch = x1_batch.cuda()
            x2_batch = x2_batch.cuda()
            y_batch = y_batch.cuda()

        preds, y_preds = model(x1_batch, x2_batch)
        loss = loss_func(preds, y_batch)
        running_losses.append(loss.item())

        y_truth = y_batch.byte()

        y_preds = torch.squeeze(y_preds, -1)
        # acc = pred.eq(truth).sum() / target.numel()
        # tmp_corrects = (classes.data == y_truth.data).sum()
        tmp_corrects = y_preds.eq(y_truth).sum()
        corrects += tmp_corrects

    size = len(x1)
    bce_loss = sum(running_losses) / len(running_losses)
    accuracy = float(corrects) / float(size)

    return bce_loss, accuracy


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
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--l2', type=float, default=0.0, metavar='LR',
                        help='l2 regularization term (default: 0.0)')
    parser.add_argument('--val-rate', type=float, default=0.1, metavar='LR',
                        help='validation rate (default: 0.1)')
    parser.add_argument('--test-rate', type=float, default=0.1, metavar='LR',
                        help='test rate (default: 0.1)')
    parser.add_argument('--embed-dim', type=int, default=128, metavar='N',
                        help='the dimensionality of the embedding space (default: 128)')
    parser.add_argument('--num-filters', type=int, default=128, metavar='N',
                        help='the number of filters (default: 128)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--one-encoder', action='store_true', default=False,
                        help='use one encoder')
    parser.add_argument('--dot-prod', action='store_true', default=False,
                        help='use dot product at the last layer')
    parser.add_argument('--model', type=str, default="cnn",
                        help='the embedding model')
    parser.add_argument('--save-dir', type=str, default="./checkpoints/",
                        help='the embedding model')
    parser.add_argument('--word-vectors', type=str, default="none",
                        help='the pre-trained word vectors')
    parser.add_argument('--multi-gpu', action='store_true', default=False,
                        help='enables multi GPU training')
    parser.add_argument('--device-id', type=int, default=0,
                        help='the gpu to be used')
    parser.add_argument('--hidden-dim', type=int, default=200, metavar='N',
                        help='the dimensionality of the hidden layer (default: 200)')
    parser.add_argument('--connect-type', type=str, default="element-wise", metavar='N',
                        help='how to connect the embedding vector and the context vector.')
    parser.add_argument('--activation', type=str, default="relu",
                        help='the activation function')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.multi_gpu = not args.no_cuda and args.multi_gpu and (torch.cuda.device_count() > 1)
    return args


def main(argv=None):
    args = parse_args()

    res = preprocess1(top=args.top, val_rate=args.val_rate,
                      test_rate=args.test_rate)
    (x1_train, x2_train, y_train,
     x1_val, x2_val, y_val,
     x1_test, x2_test, y_test, vocab1, vocab2) = res

    print("Train Pos / Neg: {} / {}".format(len(y_train[y_train == 1]),
                                            len(y_train[y_train == 0])))
    print("Val Pos / Neg: {} / {}".format(len(y_val[y_val == 1]),
                                          len(y_val[y_val == 0])))
    print("Test Pos / Neg: {} / {}".format(len(y_test[y_test == 1]),
                                           len(y_test[y_test == 0])))

    if args.word_vectors == "glove":
        word_vectors_1 = data._load_glove(vocab1, GLOVE_FILE)
        word_vectors_2 = data._load_glove(vocab2, GLOVE_FILE)
    elif args.word_vectors == "word2vec":
        word_vectors_1 = data._load_word2vec(vocab1, W2V_FILE)
        word_vectors_2 = data._load_word2vec(vocab2, W2V_FILE)
    else:
        word_vectors_1 = None
        word_vectors_2 = None

    q1_max = x1_train.shape[1]
    q2_max = x2_train.shape[1]

    if args.model == "cnn":
        # Embedding encoder
        emb_enc = CNNEncoder(vocab_size=len(vocab1.vocabulary_),
                             embed_dim=args.embed_dim,
                             word_vectors=word_vectors_1,
                             filter_sizes=FILTER_SIZES,
                             hidden_dim=args.hidden_dim,
                             num_filters=args.num_filters)
        # Context encoder
        ctx_enc = CNNEncoder(vocab_size=len(vocab1.vocabulary_),
                             embed_dim=args.embed_dim,
                             word_vectors=word_vectors_1,
                             filter_sizes=FILTER_SIZES,
                             hidden_dim=args.hidden_dim,
                             num_filters=args.num_filters)
    elif args.model == "mean":
        emb_enc = MeanEncoder(vocab_size=len(vocab1.vocabulary_),
                              embed_dim=args.embed_dim,
                              word_vectors=word_vectors_1)
        ctx_enc = MeanEncoder(vocab_size=len(vocab2.vocabulary_),
                              embed_dim=args.embed_dim,
                              word_vectors=word_vectors_2)
    elif args.model == "max":
        emb_enc = MaxEncoder(vocab_size=len(vocab1.vocabulary_),
                             embed_dim=args.embed_dim,
                             hidden_dim=args.hidden_dim,
                             word_vectors=word_vectors_1)
        ctx_enc = MaxEncoder(vocab_size=len(vocab1.vocabulary_),
                             embed_dim=args.embed_dim,
                             word_vectors=word_vectors_1)
    elif args.model == "lstm":
        emb_enc = LSTMEncoder(vocab_size=len(vocab1.vocabulary_),
                              embed_dim=args.embed_dim,
                              word_vectors=word_vectors_1,
                              hidden_dim=args.hidden_dim)
        ctx_enc = LSTMEncoder(vocab_size=len(vocab2.vocabulary_),
                              embed_dim=args.embed_dim,
                              word_vectors=word_vectors_2,
                              hidden_dim=args.hidden_dim)
    else:
        print("Wrong model!")
        return 0

    model = DualEncoders(embedding_encoder=emb_enc,
                         context_encoder=ctx_enc,
                         fc_dim=3 * args.num_filters,
                         dot_prod=args.dot_prod)

    if args.multi_gpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, CUDA_VISIBLE_DEVICES).cuda()

    elif args.cuda:
        model = model.cuda()

    # print(torch_summarize(model, show_weights=False))

    train(model, x1_train, x2_train, y_train,
          x1_val, x2_val, y_val,
          x1_test, x2_test, y_test, args)

    """
    x1_test = Variable(torch.LongTensor(x1_test))
    x2_test = Variable(torch.LongTensor(x2_test))
    if torch.cuda.is_available():
        x1_test = x1_test.cuda()
        x2_test = x2_test.cuda()

    x1_contexts = model.contexts(x1_test).mean(dim=1)
    x2_embeddings = model.embeddings(x2_test).mean(dim=1)

    print("x1_contexts.size:", x1_contexts.size())
    print("x2_embeddings.size:", x2_embeddings.size())
    print("x2_embeddings.transpose(0, 1).size:", x2_embeddings.transpose(0, 1).size())

    x = torch.matmul(x1_contexts, x2_embeddings.transpose(0, 1))

    print("x.size:", x.size())

    print("EVALUATION!")
    print("=======================")
    model.print_parameters()
    loss = eval(model, x1_dev, x2_dev, y_dev, batch_size=args.batch_size)
    print("Evaluation: loss: {}".format(loss))
    """

if __name__ == '__main__':
    main()
