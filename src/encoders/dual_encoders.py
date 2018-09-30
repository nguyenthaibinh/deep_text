import torch.nn as nn
import torch.nn.functional as F


class DualEncoders(nn.Modeule):
	def __init__(self, embedding_encoder, context_encoder, dot_prod=False):
		super(DualEncoders, self).__init__()

		self.embedding_encoder = embedding_encoder
		self.context_encoder = context_encoder

	def forward(self, x1, x2):
		h1 = self.embedding_encoder(x1)
		h2 = self.context_encoder(x2)

		dot_prod = (h1 * h2).sum(dim=1)
		preds = F.sigmoid(dot_prod)

		classes = (preds >= 0.5)

		return preds, classes
