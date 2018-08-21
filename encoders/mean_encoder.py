import torch
import torch.nn as nn
import torch.nn.functional as F


class TextNet(nn.Module):
	def __init__(self, sequence_length, vocab_size, embed_size,
	             word_vectors=None,
	             num_classes=2, l2_reg=0.0, dropout=0.5):
		super(TextNet, self).__init__()
		self.sequence_length = sequence_length
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.num_classes = num_classes
		self.l2_reg = l2_reg

		# Embedding layer
		if word_vectors is None:
			print("Use pre-trained word vectors.")
			self.embeddings = nn.Embedding(vocab_size, self.embed_size, padding_idx=0)
			nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.01)
		else:
			self.embed_size = word_vectors.shape[1]
			word_vectors = torch.FloatTensor(word_vectors)
			self.embeddings = nn.Embedding(vocab_size, self.embed_size, padding_idx=0)
			self.embeddings.weight = nn.Parameter(word_vectors, requires_grad=False)

		self.fc1 = nn.Linear(self.embed_size, self.embed_size)
		# self.fc1 = nn.Linear(self.embed_size, num_classes)
		self.fc2 = nn.Linear(self.embed_size, num_classes)
		nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
		nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)

		self.print_parameters()

	def print_parameters(self):
		print("TextNet information:")
		print("=======================================")
		print("sequence length:", self.sequence_length)
		print("vocab_size:", self.vocab_size)
		print("embed_size:", self.embed_size)
		print("num_classes:", self.num_classes)
		print("l2_reg:", self.l2_reg)

	def forward(self, x):
		emb = self.embeddings(x)

		# Apply convolution and max pooling for each filter size
		h = torch.mean(emb, dim=1)

		logits = torch.tanh(self.fc1(h))            # [B, class]

		logits = torch.tanh(self.fc2(logits))

		# logits = F.relu(self.fc2(logits))

        # Prediction
		probs = F.softmax(logits, dim=1)       # [B, class]

		classes = torch.max(probs, 1)[1]# [B]

		return probs, classes

