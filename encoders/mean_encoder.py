import torch
import torch.nn as nn
import torch.nn.functional as F


class DualMean(nn.Module):
	def __init__(self, sequence_length, vocab_size, embed_size,
	             word_vectors=None,
	             num_classes=2, l2_reg=0.0, dropout=0.5,
	             one_encoder=False, dot_prod=False):
		super(DualMean, self).__init__()
		self.sequence_length = sequence_length
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.num_classes = num_classes
		self.l2_reg = l2_reg
		self.one_encoder = one_encoder
		self.dot_prod = dot_prod

		# Embedding layer
		if word_vectors is None:
			print("Use one-hot word vectors.")
			# Create embedding and context vectors
			self.embeddings = nn.Embedding(vocab_size, self.embed_size,
			                               padding_idx=0)
			self.contexts = nn.Embedding(vocab_size, self.embed_size,
			                             padding_idx=0)

			# Initialize embedding and context vectors
			nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.01)
			nn.init.normal_(self.contexts.weight, mean=0.0, std=0.01)
		else:
			print("Use pre-trained word vectors.")
			self.embed_size = word_vectors.shape[1]
			word_vectors = torch.FloatTensor(word_vectors)

			# Create embedding and context vectors
			self.embeddings = nn.Embedding(vocab_size, self.embed_size,
			                               padding_idx=0)
			self.contexts = nn.Embedding(vocab_size, self.embed_size,
			                             padding_idx=0)

			# Load pre-trained word vectors
			self.embeddings.weight = nn.Parameter(word_vectors,
			                                      requires_grad=False)
			self.contexts.weight = nn.Parameter(word_vectors,
			                                    requires_grad=False)

		# Create embedding encoder
		self.embedding_fc1 = nn.Linear(self.embed_size, self.embed_size)
		self.embedding_fc2 = nn.Linear(self.embed_size, self.embed_size)

		nn.init.normal_(self.embedding_fc1.weight, mean=0.0, std=0.01)
		nn.init.normal_(self.embedding_fc2.weight, mean=0.0, std=0.01)

		# Create context encoder
		self.context_fc1 = nn.Linear(self.embed_size, self.embed_size)
		self.context_fc2 = nn.Linear(self.embed_size, self.embed_size)

		nn.init.normal_(self.context_fc1.weight, mean=0.0, std=0.01)
		nn.init.normal_(self.context_fc2.weight, mean=0.0, std=0.01)

		self.fc = nn.Linear(self.embed_size, 1)
		nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)

		self.embedding1_bn = nn.BatchNorm1d(self.embed_size)
		self.embedding2_bn = nn.BatchNorm1d(self.embed_size)

		self.context1_bn = nn.BatchNorm1d(self.embed_size)
		self.context2_bn = nn.BatchNorm1d(self.embed_size)

		self.print_parameters()

	def print_parameters(self):
		print("TextNet information:")
		print("=======================================")
		print("sequence length:", self.sequence_length)
		print("vocab_size:", self.vocab_size)
		print("embed_size:", self.embed_size)
		print("num_classes:", self.num_classes)
		print("l2_reg:", self.l2_reg)
		print("one_encoder:", self.one_encoder)
		print("dot_prod:", self.dot_prod)

	def embedding_encode(self, x):
		h = self.embeddings(x)
		# print("h.size:", h.size())

		# Calculate mean vector
		h = torch.mean(h, dim=1)
		h = self.embedding1_bn(h)
		h = torch.tanh(h)
		h = self.embedding2_bn(h)
		# print("h.size:", h.size())
		h = torch.tanh(self.embedding_fc1(h))
		# h = torch.tanh(self.embedding_fc2(h))

		return h

	def context_encode(self, x):
		h = self.contexts(x)
		# print("h.size:", h.size())

		# Calculate mean vector
		h = torch.mean(h, dim=1)
		h = self.context1_bn(h)

		h = torch.tanh(h)
		h = self.context2_bn(h)
		# print("h.size:", h.size())
		h = torch.tanh(self.context_fc1(h))
		# h = torch.tanh(self.context_fc2(h))

		return h

	def forward(self, x1, x2):
		h1 = self.context_encode(x1)

		if self.one_encoder is True:
			h2 = self.context_encode(x2)
		else:
			h2 = self.embedding_encode(x2)

		if self.dot_prod is True:
			dot_prod = (h1 * h2).sum(dim=1)
			preds = F.sigmoid(dot_prod)
		else:
			dot_prod = self.fc(h1 * h2)
			preds = F.sigmoid(dot_prod)

		classes = (preds >= 0.5)

		return preds, classes
