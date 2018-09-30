
import torch
import torch.nn as nn
import torch.nn.functional as F


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
