import torch
import torch.nn as nn


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
