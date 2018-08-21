import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
	def __init__(self, sequence_length, vocab_size, embed_size,
	             filter_sizes, num_filters, word_vectors=None,
	             num_classes=2, l2_reg=0.0, dropout=0.5,):
		super(CNNEncoder, self).__init__()
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
			self.embeddings = nn.Embedding(vocab_size, self.embed_size, padding_idx=0)
			nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.01)
		else:
			print("Use pre-trained word vectors.")
			self.embed_size = word_vectors.shape[1]
			word_vectors = torch.FloatTensor(word_vectors)
			self.embeddings = nn.Embedding(vocab_size, self.embed_size, padding_idx=0)
			self.embeddings.weight = nn.Parameter(word_vectors, requires_grad=False)

		self.convs = nn.ModuleList([nn.Conv1d(in_channels=self.embed_size,
		                                      out_channels=num_filters,
		                                      kernel_size=filter_size)
		                           for filter_size in filter_sizes])

		self.dropout = nn.Dropout(dropout)

		self.fc1 = nn.Linear(num_filters * len(filter_sizes), num_filters * len(filter_sizes))
		# self.fc1 = nn.Linear(num_filters * len(filter_sizes), num_classes)
		self.fc2 = nn.Linear(num_filters * len(filter_sizes), num_classes)
		nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
		nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)

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

	def forward(self, x):
		emb = self.embeddings(x)
		emb = emb.transpose(1, 2)

		feature_list = []
		for conv in self.convs:
			h = torch.tanh(conv(emb))
			h = F.max_pool1d(h, h.size(2))
			feature_list.append(h)

		h = torch.cat(feature_list, dim=1)
		h = torch.squeeze(h, -1)
		h = self.dropout(h)

		logits = torch.tanh(self.fc1(h))
		logits = torch.tanh(self.fc2(logits))

        # Prediction
		probs = F.softmax(logits, dim=1)       # [B, class]

		classes = torch.max(probs, 1)[1]# [B]

		return probs, classes