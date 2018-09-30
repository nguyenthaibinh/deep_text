# import email
import re
import time
import numpy as np
from pprint import pprint
import torch
from torch.nn.modules.module import _addindent

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


def _write_elapsed_time(start_t=None, msg=None):
    cur_time = time.time()
    if start_t is None:
        print("{}".format(msg))
    else:
        elapsed_time = time.strftime("%H:%M:%S",
                                     time.gmtime(cur_time - start_t))
        print("{}".format(msg), elapsed_time)
    return cur_time


def negative_sampling(n_items, type="normal", repeat=True):
    neg = np.random.randint(n_items)
    return neg


def get_train_examples(train_requests, train_responses, n_items, n_neg,
                       verbose=False, weights=None):
    start_t = time.time()
    requests, responses, labels = [], [], []
    n_pairs = len(train_requests)

    if verbose:
        print("n_pairs:", n_pairs, "n_neg:", n_neg)
    for i in range(len(train_requests)):
        # positive instance
        requests.append(train_requests[i])
        responses.append(train_responses[i])
        labels.append(1)

        # negative instances
        for t in range(n_neg):
            j = negative_sampling(n_items=n_items)
            """
            try:
                while train.has_key((u, j)):
                    j = np.random.randint(n_items)
            except:
                while (u, j) in train.keys():
                    j = np.random.randint(n_items)
            """
            requests.append(train_requests[i])
            responses.append(j)
            labels.append(0)
    if verbose:
        _write_elapsed_time(start_t, "{} train instances obtained:".format(len(requests)))
    requests = np.array(requests, dtype=np.int64)
    responses = np.array(responses, dtype=np.int64)
    labels = np.array(labels, dtype=np.float32)
    return requests, responses, labels


def standardize_sentence(s):
    """
    - Remove all non alphabet, dot, colon, semi-colon of a given string.
    - Remove double spaces
    """
    ret = 0
    return ret


def write_vocab(vocab, file_name):
    f = open(file_name, "w")
    pprint(vocab.vocabulary_._mapping, f)
    f.close()


def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if w not in stops]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return(text)


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr
