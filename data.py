import pandas as pd
import numpy as np
from os.path import dirname, abspath, join, exists
from gensim.models import KeyedVectors

QUORA_DATA_FILE = "./data/quora/quora_duplicate_questions.tsv"

BASE_DIR = dirname(abspath(__file__))
DATA_DIR = join(BASE_DIR, 'data')


def load_quora_data(top=0):
	df = pd.read_csv(QUORA_DATA_FILE, sep='\t')
	df = df.dropna()

	n_len = df.shape[0]
	print("n_len:", n_len)

	if isinstance(top, int) and (top > 0) and (top < n_len):
		print("Get top {} rows of the quora.".format(top))
		df = df[:top]

	text1 = list(df['question1'])
	text2 = list(df['question2'])
	is_duplicate = list(df['is_duplicate'])

	return text1, text2, is_duplicate


def batch_iter(data, batch_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    # Shuffle the data
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]


def _load_word2vec(vocab_processor, wordvector_file):
    if not exists(wordvector_file):
        raise Exception("You must download word vectors through `download_wordvec.py` first")
    word2vec = KeyedVectors.load_word2vec_format(wordvector_file, binary=True)
    vector_size = word2vec.vector_size

    vocab_dict = vocab_processor.vocabulary_._mapping
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    vocabulary = list(list(zip(*sorted_vocab))[0])

    word_vectors = []
    for word in vocabulary:
        if word in word2vec.vocab:
            vector = word2vec[word]
        else:
            vector = np.random.normal(scale=0.2, size=vector_size) # random vector

        word_vectors.append(vector)

    weight = np.stack(word_vectors)
    return weight


def _load_glove(vocab_processor, wordvector_file):
    vector_size = 300
    if not exists(wordvector_file):
        raise Exception("You must download word vectors through `download_wordvec.py` first")

    glove_model = {}
    with open(wordvector_file) as file:
        for line in file:
            line_split = line.split()
            word = ' '.join(line_split[:-vector_size])
            numbers = line_split[-vector_size:]
            glove_model[word] = numbers
    glove_vocab = glove_model.keys()

    vocab_dict = vocab_processor.vocabulary_._mapping
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    vocabulary = list(list(zip(*sorted_vocab))[0])

    word_vectors = []
    for word in vocabulary:

        if word in glove_vocab:
            vector = np.array(glove_model[word], dtype=float)
        else:
            vector = np.random.normal(scale=0.2, size=vector_size) # random vector

        word_vectors.append(vector)

    weight = np.stack(word_vectors)
    return weight
