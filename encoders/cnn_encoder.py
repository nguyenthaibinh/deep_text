import torch
import torch.nn as nn
import torch.nn.functional as F


class DualCNN(nn.Module):
	def __init__(self, sequence_length, vocab_size, embed_size,
	             filter_sizes, num_filters, word_vectors=None,
	             num_classes=2, l2_reg=0.0, dropout=0.5):
		super(DualCNN, self).__init__()
		self.sequence_length = sequence_length
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.filter_sizes = filter_sizes
		self.num_filters = num_filters
		self.num_classes = num_classes
		self.l2_reg = l2_reg

		# Embedding layer
		if word_vectors is None:
			print("Use one-hot word vectors.")
			# Create embedding and context vectors
			self.embeddings = nn.Embedding(vocab_size, self.embed_size, padding_idx=0)
			self.contexts = nn.Embedding(vocab_size, self.embed_size, padding_idx=0)

			# Initialize embedding and context vectors
			nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.01)
			nn.init.normal_(self.contexts.weight, mean=0.0, std=0.01)
		else:
			print("Use pre-trained word vectors.")
			self.embed_size = word_vectors.shape[1]
			word_vectors = torch.FloatTensor(word_vectors)

			# Create embedding and context vectors
			self.embeddings = nn.Embedding(vocab_size, self.embed_size, padding_idx=0)
			self.contexts = nn.Embedding(vocab_size, self.embed_size, padding_idx=0)

			# Load pre-trained word vectors
			self.embeddings.weight = nn.Parameter(word_vectors, requires_grad=False)
			self.contexts.weight = nn.Parameter(word_vectors, requires_grad=False)

		# Create embedding encoder
		self.embedding_encoder = nn.ModuleList(
		                        [nn.Conv1d(in_channels=self.embed_size,
		                                   out_channels=num_filters,
		                                   kernel_size=filter_size)
		                        for filter_size in filter_sizes])

		self.embedding_fc1 = nn.Linear(num_filters * len(filter_sizes),
		                               num_filters * len(filter_sizes))
		self.embedding_fc2 = nn.Linear(num_filters * len(filter_sizes),
		                               num_filters * len(filter_sizes))

		nn.init.normal_(self.embedding_fc1.weight, mean=0.0, std=0.01)
		nn.init.normal_(self.embedding_fc2.weight, mean=0.0, std=0.01)

		# Create context encoder
		self.context_encoder = nn.ModuleList(
		                        [nn.Conv1d(in_channels=self.embed_size,
		                                   out_channels=num_filters,
		                                   kernel_size=filter_size)
		                        for filter_size in filter_sizes])

		self.context_fc1 = nn.Linear(num_filters * len(filter_sizes),
		                             num_filters * len(filter_sizes))
		self.context_fc2 = nn.Linear(num_filters * len(filter_sizes),
		                             num_filters * len(filter_sizes))

		nn.init.normal_(self.context_fc1.weight, mean=0.0, std=0.01)
		nn.init.normal_(self.context_fc2.weight, mean=0.0, std=0.01)

		# Create dropout layer
		self.embedding_dropout = nn.Dropout(dropout)
		self.context_dropout = nn.Dropout(dropout)

		self.print_parameters()

	def print_parameters(self):
		print("TextCNN-1D information:")
		print("=======================================")
		print("sequence length:", self.sequence_length)
		print("vocab_size:", self.vocab_size)
		print("embed_size:", self.embed_size)
		print("filter_sizes:", self.filter_sizes)
		print("num_filters:", self.num_filters)
		print("num_classes:", self.num_classes)
		print("l2_reg:", self.l2_reg)

	def embedding_encode(self, x):
		h = self.embeddings(x)
		h = h.transpose(1, 2)

		feature_list = []
		for conv in self.convs:
			h = torch.tanh(conv(h))
			h = F.max_pool1d(h, h.size(2))
			feature_list.append(h)

		h = torch.cat(feature_list, dim=1)
		h = torch.squeeze(h, -1)
		h = self.dropout(h)

		logits = torch.tanh(self.embedding_fc1(h))
		logits = torch.tanh(self.embedding_fc2(logits))

	def context_encode(self, x):
		h = self.contexts(x)
		h = h.transpose(1, 2)

		feature_list = []
		for conv in self.convs:
			h = torch.tanh(conv(h))
			h = F.max_pool1d(h, h.size(2))
			feature_list.append(h)

		h = torch.cat(feature_list, dim=1)
		h = torch.squeeze(h, -1)
		h = self.dropout(h)

		logits = torch.tanh(self.context_fc1(h))
		logits = torch.tanh(self.context_fc2(logits))

	def forward(self, x1, x2):
		h1 = self.embedding_encode(x1)
		h2 = self.context_encode(x2)

		dot_prod = (h1 * h2).sum(dim=1)

		preds = F.sigmoid(dot_prod)

		return preds
