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
