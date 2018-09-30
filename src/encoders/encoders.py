import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
	def __init__(self, vocab_size, embed_dim,
				 filter_sizes, num_filters, hidden_dim=200,
				 word_vectors=None, dropout=0.2):
		super(CNNEncoder, self).__init__()
		self.vocab_size = vocab_size
		self.embed_dim = embed_dim
		self.filter_sizes = filter_sizes
		self.num_filters = num_filters

		# Embedding layer
		if word_vectors is None:
			print("Use one-hot word vectors.")
			# Create embedding and context vectors
			self.embeddings = nn.Embedding(vocab_size + 1, self.embed_dim,
											padding_idx=0)

			# Initialize embedding and context vectors
			nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.01)
		else:
			print("Use pre-trained word vectors.")
			self.embed_dim = word_vectors.shape[1]
			word_vectors = torch.FloatTensor(word_vectors)

			# Create embedding and context vectors
			self.embeddings = nn.Embedding(vocab_size + 1, self.embed_dim,
										   padding_idx=0)

			# Load pre-trained word vectors
			self.embeddings.weight = nn.Parameter(word_vectors,
			                                      requires_grad=False)

		# Convolutional layers
		self.convs = nn.ModuleList(
						[nn.Conv1d(in_channels=self.embed_dim,
								   out_channels=num_filters,
								   kernel_size=filter_size)
						for filter_size in filter_sizes])

		# Fully connected layers
		fc_dim = num_filters * len(filter_sizes)
		if hidden_dim is None:
			self.enc_dim = num_filters * len(filter_sizes)
		else:
			self.enc_dim = hidden_dim

		self.fc = nn.Sequential(
					# nn.BatchNorm1d(fc_dim),
					nn.Dropout(dropout),
					nn.Linear(fc_dim, self.enc_dim),
					nn.Tanh()
					# nn.Dropout(dropout)
					)

	def forward(self, x):
		features = self.embeddings(x)
		features = features.transpose(1, 2)

		feature_list = []
		for conv in self.convs:
			h = torch.tanh(conv(features))
			h = F.max_pool1d(h, h.size(2))
			feature_list.append(h)

		h = torch.cat(feature_list, dim=1)
		h = torch.squeeze(h, -1)
		h = self.fc(h)

		return h


class MeanEncoder(nn.Module):
	def __init__(self, vocab_size, embed_dim, word_vectors=None,
				 hidden_dim=200, dropout=0.2):
		super(MeanEncoder, self).__init__()
		self.vocab_size = vocab_size
		self.embed_dim = embed_dim

		# Embedding layer
		if word_vectors is None:
			print("Use one-hot word vectors.")
			# Create embedding and context vectors
			self.embeddings = nn.Embedding(vocab_size + 1, self.embed_dim,
										   padding_idx=0)

			# Initialize embedding and context vectors
			nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.01)
		else:
			print("Use pre-trained word vectors.")
			self.embed_dim = word_vectors.shape[1]
			word_vectors = torch.FloatTensor(word_vectors)

			# Create embedding and context vectors
			self.embeddings = nn.Embedding(vocab_size + 1, self.embed_dim,
										   padding_idx=0)

			# Load pre-trained word vectors
			self.embeddings.weight = nn.Parameter(word_vectors,
												  requires_grad=False)

		# Fully connected layers
		if hidden_dim is None:
			self.enc_dim = self.embed_dim
		else:
			self.enc_dim = hidden_dim

		# Fully connected layers
		self.fc_layers = nn.Sequential(
							nn.BatchNorm1d(self.embed_dim),
							nn.Linear(self.embed_dim, self.enc_dim),
							nn.ReLU(),
							nn.Dropout(dropout),

							nn.BatchNorm1d(self.enc_dim),
							nn.Linear(self.enc_dim, self.enc_dim),
							nn.ReLU(),
							nn.Dropout(dropout),

							nn.BatchNorm1d(self.enc_dim),
							nn.Linear(self.enc_dim, self.enc_dim),
							nn.ReLU(),
							nn.Dropout(dropout)
							)

	def forward(self, x):
		h = self.embeddings(x)
		# print("h.size:", h.size())

		# Calculate mean vector
		h = torch.mean(h, dim=1)
		h = self.fc_layers(h)

		return h


class MaxEncoder(nn.Module):
	def __init__(self, vocab_size, embed_dim, word_vectors=None,
				 hidden_dim=200, dropout=0.2):
		super(MaxEncoder, self).__init__()
		self.vocab_size = vocab_size
		self.embed_dim = embed_dim

		# Embedding layer
		if word_vectors is None:
			print("Use one-hot word vectors.")
			# Create embedding and context vectors
			self.embeddings = nn.Embedding(vocab_size + 1, self.embed_dim,
										   padding_idx=0)

			# Initialize embedding and context vectors
			nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.01)
		else:
			print("Use pre-trained word vectors.")
			self.embed_dim = word_vectors.shape[1]
			word_vectors = torch.FloatTensor(word_vectors)

			# Create embedding and context vectors
			self.embeddings = nn.Embedding(vocab_size + 1, self.embed_dim,
										   padding_idx=0)

			# Load pre-trained word vectors
			self.embeddings.weight = nn.Parameter(word_vectors,
												  requires_grad=False)

		# Fully connected layers
		if hidden_dim is None:
			self.enc_dim = self.embed_dim
		else:
			self.enc_dim = hidden_dim

		# Fully connected layers
		self.fc_layers = nn.Sequential(
							nn.BatchNorm1d(self.embed_dim),
							nn.Linear(self.embed_dim, self.enc_dim),
							nn.ReLU(),
							nn.Dropout(dropout),

							nn.BatchNorm1d(self.enc_dim),
							nn.Linear(self.enc_dim, self.enc_dim),
							nn.ReLU(),
							nn.Dropout(dropout),

							nn.BatchNorm1d(self.enc_dim),
							nn.Linear(self.enc_dim, self.enc_dim),
							nn.ReLU(),
							nn.Dropout(dropout)
							)

		self.enc_dim = self.embed_dim

	def forward(self, x):
		h = self.embeddings(x)
		# print("h.size:", h.size())

		# Calculate mean vector
		h = torch.max(h, dim=1)
		h = self.fc_layers(h)

		return h


class LSTMEncoder(nn.Module):
	def __init__(self, vocab_size, embed_dim, hidden_dim=300,
				 word_vectors=None, dropout=0.2):
		super(LSTMEncoder, self).__init__()
		self.embed_dim = embed_dim
		self.hidden_dim = hidden_dim

		# Embedding layer
		if word_vectors is None:
			print("Use one-hot word vectors.")
			# Create embedding and context vectors
			self.embeddings = nn.Embedding(vocab_size + 1, self.embed_dim,
			                               padding_idx=0)

			# Initialize embedding and context vectors
			nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.01)
		else:
			print("Use pre-trained word vectors.")
			self.embed_dim = word_vectors.shape[1]
			word_vectors = torch.FloatTensor(word_vectors)

			# Create embedding and context vectors
			self.embeddings = nn.Embedding(vocab_size + 1, self.embed_dim,
										   padding_idx=0)

			# Load pre-trained word vectors
			self.embeddings.weight = nn.Parameter(word_vectors,
												  requires_grad=False)

		self.rnn = nn.LSTM(self.embed_dim, self.hidden_dim)

		self.enc_dim = self.hidden_dim

		# Fully connected layers
		self.fc_layers = nn.Sequential(
							# nn.BatchNorm1d(self.enc_dim),
							nn.Linear(self.enc_dim, self.enc_dim)
							# nn.ReLU()
							# nn.Dropout(dropout)

							# nn.BatchNorm1d(self.hidden_dim),
							# nn.Linear(self.hidden_dim, self.hidden_dim),
							# nn.ReLU(),
							# nn.Dropout(dropout),

							# nn.BatchNorm1d(self.hidden_dim),
							# nn.Linear(self.hidden_dim, self.hidden_dim),
							# nn.ReLU(),
							# nn.Dropout(dropout)
							)

	def forward(self, x):
		embeds = self.embeddings(x)
		h, (h0, c0) = self.rnn(embeds)
		h = h.mean(dim=1)
		h = h.contiguous()
		# h = h.view(-1, h.shape[2])
		# print("h.view(-1, h.shape[2]).size:", h.size())
		return h
