import torch.nn as nn
import torch


class DualEncoders(nn.Module):
	def __init__(self, embedding_encoder, context_encoder,
	             fc_dim=200, dot_prod=False):
		super(DualEncoders, self).__init__()

		self.embedding_encoder = embedding_encoder
		self.context_encoder = context_encoder

		self.dot_prod = dot_prod

		fc_dim = self.embedding_encoder.enc_dim

		self.fc_block = nn.Sequential(
		                nn.BatchNorm1d(fc_dim),
		                nn.Linear(fc_dim, 1))

	def forward(self, x1, x2):
		h1 = self.embedding_encoder(x1)
		h2 = self.context_encoder(x2)

		if self.dot_prod is True:
			dot_prod = (h1 * h2).sum(dim=1)
		else:
			prod = h1 * h2
			dot_prod = self.fc_block(prod)

		preds = torch.sigmoid(dot_prod)

		pred_labels = (preds >= 0.5)

		return preds, pred_labels
