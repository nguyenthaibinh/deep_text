import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
	def __init__(self, vocab_size, embed_size,
	             filter_sizes, num_filters, word_vectors=None,
	             dropout=0.2):
		super(CNNEncoder, self).__init__()
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.filter_sizes = filter_sizes
		self.num_filters = num_filters

		# Embedding layer
		if word_vectors is None:
			print("Use one-hot word vectors.")
			# Create embedding and context vectors
			self.embeddings = nn.Embedding(vocab_size + 1, self.embed_size,
											padding_idx=0)

			# Initialize embedding and context vectors
			nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.01)
		else:
			print("Use pre-trained word vectors.")
			self.embed_size = word_vectors.shape[1]
			word_vectors = torch.FloatTensor(word_vectors)

			# Create embedding and context vectors
			self.embeddings = nn.Embedding(vocab_size + 1, self.embed_size,
			                               padding_idx=0,
			                               requires_grad=False)

		# Convolutional layers
		self.convs = nn.ModuleList(
                        [nn.Conv1d(in_channels=self.embed_size,
                                   out_channels=num_filters,
                                   kernel_size=filter_size)
                        for filter_size in filter_sizes])
		"""
		self.convs = nn.ModuleList(
		                [nn.Sequential(nn.Conv1d(in_channels=self.embed_size,
		                                         out_channels=num_filters,
		                                         kernel_size=filter_size),
		                				nn.ReLU()
		                				# nn.MaxPool1d(kernel_size=filter_size),
		                				)
		                for filter_size in filter_sizes])
		"""
		"""
		convs = []
		for filter_size in filter_sizes:
			conv_module = nn.Sequential(
						nn.Conv1d(in_channels=self.embed_size,
						          out_channels=num_filters,
						          kernel_size=filter_size),
						nn.MaxPool1d(kernel_size=filter_size),
						nn.ReLU())
			convs.append(conv_module)
		self.convs = nn.Sequential(convs)
		"""

		"""
		self.convs = torch.nn.Sequential()
		for i, filter_size in enumerate(filter_sizes):
			self.convs.add_module("conv_1_filter_{}".format(i + 1),
			                      nn.Conv1d(in_channels=self.embed_size,
			                                out_channels=num_filters,
			                                kernel_size=filter_size))
			# self.convs.add_module("dropout_filter_{}".format(i + 1),
			#                      nn.Dropout(dropout))
			self.convs.add_module("maxpool_1_filter_{}".format(i + 1),
			                      nn.MaxPool1d(kernel_size=filter_size))
			self.convs.add_module("relu_1_filter_{}".format(i + 1),
			                      torch.nn.ReLU())
		"""

		# Fully connected layers
		fc_dim = num_filters * len(filter_sizes)
		self.fc = nn.Sequential(
		          	nn.BatchNorm1d(fc_dim),
		          	nn.Linear(fc_dim, fc_dim),
		          	nn.ReLU(),
		          	nn.Dropout(dropout)
		            )

		self.out_dim = num_filters * len(filter_sizes)

	def forward(self, x):
		features = self.embeddings(x)
		features = features.transpose(1, 2)

		feature_list = []
		for conv in self.convs:
			h = torch.relu(conv(features))
			h = F.max_pool1d(h, h.size(2))
			feature_list.append(h)

		h = torch.cat(feature_list, dim=1)
		h = torch.squeeze(h, -1)
		h = self.fc(h)

		return h


class DualCNN(nn.Module):
	def __init__(self, vocab1_size, vocab2_size,
	             embed_size, filter_sizes, num_filters,
	             word_vectors_1=None, word_vectors_2=None,
	             num_classes=2, l2_reg=0.0, dropout=0.1,
	             one_encoder=False, dot_prod=False):
		super(DualCNN, self).__init__()
		self.vocab1_size = vocab1_size
		self.vocab2_size = vocab2_size
		self.embed_size = embed_size
		self.filter_sizes = filter_sizes
		self.num_filters = num_filters
		self.num_classes = num_classes
		self.l2_reg = l2_reg
		self.one_encoder = one_encoder
		self.dot_prod = dot_prod

		# Embedding layer
		if word_vectors_1 is None:
			print("Use one-hot word vectors.")
			# Create embedding and context vectors
			self.embeddings = nn.Embedding(vocab1_size + 1, self.embed_size,
			                               padding_idx=0)
			self.contexts = nn.Embedding(vocab2_size + 1, self.embed_size,
			                             padding_idx=0)

			# Initialize embedding and context vectors
			nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.01)
			nn.init.normal_(self.contexts.weight, mean=0.0, std=0.01)
		else:
			print("Use pre-trained word vectors.")
			self.embed_size = word_vectors_1.shape[1]
			word_vectors_1 = torch.FloatTensor(word_vectors_1)
			word_vectors_2 = torch.FloatTensor(word_vectors_2)

			# Create embedding and context vectors
			self.embeddings = nn.Embedding(vocab1_size + 1, self.embed_size,
			                               padding_idx=0)
			self.contexts = nn.Embedding(vocab2_size + 1, self.embed_size,
			                             padding_idx=0)

			# Load pre-trained word vectors
			self.embeddings.weight = nn.Parameter(word_vectors_1,
			                                      requires_grad=False)
			self.contexts.weight = nn.Parameter(word_vectors_2,
			                                    requires_grad=False)

		# Create embedding encoder
		self.embedding_convs = nn.ModuleList(
		                        [nn.Conv1d(in_channels=self.embed_size,
		                                   out_channels=num_filters,
		                                   kernel_size=filter_size)
		                        for filter_size in filter_sizes])

		# Create context encoder
		self.embedding_modules = nn.ModuleList(
		                        [nn.Sequential(nn.Conv1d(in_channels=self.embed_size,
		                                                 out_channels=num_filters,
		                                                 kernel_size=filter_size),
		                        			   nn.MaxPool1d(kernel_size=filter_size),
		                        			   nn.ReLU())
		                        for filter_size in filter_sizes])

		self.embedding_fc1 = nn.Linear(num_filters * len(filter_sizes),
		                               num_filters * len(filter_sizes))
		self.embedding_fc2 = nn.Linear(num_filters * len(filter_sizes),
		                               num_filters * len(filter_sizes))

		nn.init.normal_(self.embedding_fc1.weight, mean=0.0, std=0.01)
		nn.init.normal_(self.embedding_fc2.weight, mean=0.0, std=0.01)

		# Create context encoder
		self.context_convs = nn.ModuleList(
		                        [nn.Conv1d(in_channels=self.embed_size,
		                                   out_channels=num_filters,
		                                   kernel_size=filter_size)
		                        for filter_size in filter_sizes])

		# Create context encoder
		self.context_modules = nn.ModuleList(
		                        [nn.Sequential(nn.Conv1d(in_channels=self.embed_size,
		                                                 out_channels=num_filters,
		                                                 kernel_size=filter_size),
		                        			   nn.MaxPool1d(kernel_size=filter_size),
		                        			   nn.ReLU())
		                        for filter_size in filter_sizes])

		self.context_fc1 = nn.Linear(num_filters * len(filter_sizes),
		                             num_filters * len(filter_sizes))
		self.context_fc2 = nn.Linear(num_filters * len(filter_sizes),
		                             num_filters * len(filter_sizes))

		nn.init.normal_(self.context_fc1.weight, mean=0.0, std=0.01)
		nn.init.normal_(self.context_fc2.weight, mean=0.0, std=0.01)

		# Create dropout layer
		self.embedding_dropout1 = nn.Dropout(dropout)
		self.embedding_dropout2 = nn.Dropout(dropout)
		self.context_dropout1 = nn.Dropout(dropout)
		self.context_dropout2 = nn.Dropout(dropout)

		self.fc = nn.Linear(num_filters * len(filter_sizes), 1)
		nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)

		self.embedding1_bn = nn.BatchNorm1d(num_filters * len(filter_sizes),
		                                    affine=False)
		self.embedding2_bn = nn.BatchNorm1d(num_filters * len(filter_sizes),
		                                    affine=False)

		self.context1_bn = nn.BatchNorm1d(num_filters * len(filter_sizes),
		                                  affine=False)
		self.context2_bn = nn.BatchNorm1d(num_filters * len(filter_sizes),
		                                  affine=False)

		# self.print_parameters()

	def print_parameters(self):
		print("TextCNN-1D information:")
		print("=======================================")
		print("vocab1_size:", self.vocab1_size)
		print("vocab2_size:", self.vocab2_size)
		print("embed_size:", self.embed_size)
		print("filter_sizes:", self.filter_sizes)
		print("num_filters:", self.num_filters)
		print("num_classes:", self.num_classes)
		print("l2_reg:", self.l2_reg)
		print("one_encoder:", self.one_encoder)
		print("dot_prod:", self.dot_prod)

	def embedding_encode(self, x):
		features = self.embeddings(x)
		features = features.transpose(1, 2)

		feature_list = []

		"""
		for module in self.embedding_modules:
			h = module(features)
			feature_list.append(h)
		"""
		for conv in self.embedding_convs:
			h = F.relu(conv(features))
			h = F.max_pool1d(h, h.size(2))
			feature_list.append(h)

		h = torch.cat(feature_list, dim=1)
		h = torch.squeeze(h, -1)
		h = self.embedding_dropout1(h)
		h = self.embedding1_bn(h)
		# h = self.embedding_fc1(h)
		h = torch.relu(self.embedding_fc1(h))
		h = self.embedding_dropout2(h)
		h = self.embedding2_bn(h)
		# h = torch.tanh(self.embedding_fc2(h))

		return h

	def context_encode(self, x):
		features = self.contexts(x)
		features = features.transpose(1, 2)

		feature_list = []

		"""
		for module in self.context_modules:
			h = module(features)
			feature_list.append(h)

		"""
		for conv in self.context_convs:
			h = F.relu(conv(features))
			h = F.max_pool1d(h, h.size(2))
			feature_list.append(h)

		h = torch.cat(feature_list, dim=1)
		h = torch.squeeze(h, -1)
		h = self.context_dropout1(h)
		h = self.context1_bn(h)
		# h = self.context_fc1(h)
		h = torch.relu(self.context_fc1(h))
		h = self.context_dropout2(h)
		h = self.context2_bn(h)
		# h = torch.tanh(self.context_fc2(h))

		return h

	def forward(self, x1, x2):
		h1 = self.embedding_encode(x1)

		if self.one_encoder is True:
			h2 = self.embedding_encode(x2)
		else:
			h2 = self.context_encode(x2)

		if self.dot_prod is True:
			dot_prod = (h1 * h2).sum(dim=1)
			# preds = F.sigmoid(dot_prod)
		else:
			dot_prod = self.fc(h1 * h2)
			# preds = torch.sigmoid(dot_prod)

		preds = torch.sigmoid(dot_prod)

		classes = (preds >= 0.5)

		return preds, classes
