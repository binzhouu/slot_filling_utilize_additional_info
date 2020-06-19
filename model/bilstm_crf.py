# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-06-19
Description: 
"""
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from model.crf import CRF
from utils.functions import build_pretrain_embedding


class BilstmCrf(nn.Module):
	def __init__(self, data, model_config):
		super(BilstmCrf, self).__init__()
		if model_config['random_embedding'] == 'True':
			self.char_embeddings = nn.Embedding(data.char_alphabet_size, model_config['char_emb_dim'])
			self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(
				data.char_alphabet_size, model_config['char_emb_dim'])))
			self.char_drop = nn.Dropout(model_config['dropout'])
		else:
			char_emb_path = model_config['char_emb_file']
			self.pretrain_char_embedding, self.char_emb_dim = build_pretrain_embedding(char_emb_path, data.char_alphabet)
			self.char_embeddings = nn.Embedding(data.char_alphabet_size, model_config['char_emb_dim'])
			self.char_embeddings.weight.data.copy_(torch.from_numpy(self.pretrain_char_embedding))
			# set 'inf' to 0:
			self.char_embeddings.weight.data[0] = torch.zeros(200)
			self.char_drop = nn.Dropout(model_config['dropout'])

		self.intent_embeddings = nn.Embedding(data.intent_alphabet_size, model_config['intent_emb_dim'])
		self.intent_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(
			data.intent_alphabet_size, model_config['intent_emb_dim'])))

		self.input_drop = nn.Dropout(model_config['dropout'])

		self.lstm = nn.LSTM(
			model_config['char_emb_dim'] + model_config['intent_emb_dim'], model_config['lstm_hidden_dim'] // 2,
			num_layers=model_config['num_layers'], batch_first=model_config['batch_first'],
			bidirectional=model_config['bidirectional'])
		self.drop_lstm = nn.Dropout(model_config['dropout'])

		self.hidden2tag = nn.Linear(model_config['lstm_hidden_dim'], data.label_alphabet_size + 2)
		self.crf = CRF(data.label_alphabet_size, model_config['gpu'])

		self.num_layers = model_config['num_layers']
		self.hidden_size = model_config['lstm_hidden_dim'] // 2
		self.device = model_config['device']

	def forward(self, batch_char, batch_intent, batch_char_len, mask, batch_label=None):
		char_embeds = self.char_embeddings(batch_char)
		intent_embeds = self.intent_embeddings(batch_intent)
		intent_embeds = torch.repeat_interleave(intent_embeds, batch_char.size(1), dim=1)

		input_embeds = torch.cat([char_embeds, intent_embeds], 2)
		input_represent = self.input_drop(input_embeds)

		# 不采用动态Rnn
		h0 = torch.zeros(self.num_layers * 2, batch_char.size(0), self.hidden_size).to(self.device)
		c0 = torch.zeros(self.num_layers * 2, batch_char.size(0), self.hidden_size).to(self.device)
		lstm_out, _ = self.lstm(input_represent, (h0, c0))

		outputs = self.hidden2tag(lstm_out)

		if batch_label is not None:
			total_loss = self.crf.neg_log_likelihood_loss(outputs, mask, batch_label)
			scores, tag_seq = self.crf._viterbi_decode(outputs, mask)
			return total_loss, tag_seq
		else:
			scores, tag_seq = self.crf._viterbi_decode(outputs, mask)
			return tag_seq

	@staticmethod
	def random_embedding(vocab_size, embedding_dim):
		pretrain_emb = np.empty([vocab_size, embedding_dim])
		scale = np.sqrt(3.0 / embedding_dim)
		for index in range(vocab_size):
			pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
		return pretrain_emb
