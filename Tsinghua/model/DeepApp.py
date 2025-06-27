# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
# ############# simple dnn model ####################### #
class AppPreGtm(nn.Module):
	"""baseline rnn model, only use time, APP usage"""

	def __init__(self, parameters):
		super(AppPreGtm, self).__init__()
		self.tim_size = parameters.tim_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.app_emd_mode = parameters.app_emb_mode


		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)

		input_size = 24 + self.uid_emb_size
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.dec_app = nn.Linear(input_size, self.app_size)

	def forward(self, tim, app, loc, uid, ptim):
		ptim_emb = self.emb_tim(ptim).squeeze(1)
		uid_emb = self.emb_uid(uid).repeat(tim.size()[0], 1)
		x = torch.cat((ptim_emb,uid_emb), 1)
		x = self.dropout(x)
		out = self.dec_app(x)
		score = F.sigmoid(out)

		return score

class AppPreUserGtm(nn.Module):
	"""baseline rnn model, only use time, APP usage"""

	def __init__(self, app_size, uid_size, hid_dim, dropout_p=0.1):
		super(AppPreUserGtm, self).__init__()
		self.app_size = app_size
		self.app_emb_size = hid_dim
		self.uid_size = uid_size
		self.uid_emb_size = hid_dim
		self.dropout_p = dropout_p

		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)

		input_size = 24 + self.app_emb_size + self.uid_emb_size
		self.dropout = nn.Dropout(p=dropout_p)
		self.dec_app = nn.Linear(input_size, self.app_size)

	def forward(self, tim, app, uid, targets, mode):
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		

		
		app_emb = app_emb
		uid_emb = self.emb_uid(uid)

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		tim = tim.to(device)
		app_emb = app_emb.to(device)

		uid_emb = uid_emb.to(device)


		x = torch.cat((tim.squeeze(1), app_emb, uid_emb.squeeze(1)), dim=1)  # shape: [B, L_total, D] or [B, D_total]

		#x = torch.cat((tim, app_emb, uid_emb), 1)
		x = self.dropout(x)
		score = self.dec_app(x)

		if mode == 'predict':
			loss = torch.mean(F.cross_entropy(score, targets.view(-1), reduction='none'))
			return score
		else:
			loss = torch.mean(F.cross_entropy(score, targets.view(-1), reduction='none'))
			return loss
