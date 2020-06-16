# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-06-10
Description: 
"""
import torch
import torch.nn as nn
import numpy as np
# from torch.nn.utils.rnn import pack_padded_sequence
# from torch.nn.utils.rnn import pad_packed_sequence
from model.crf import CRF
from model.self_attn import ScaledDotProductAttention
from utils.functions import build_pretrain_embedding, get_attn_key_pad_mask


class CnnAttnLstmCRF(nn.Module):
	def __init__(self, data, model_config):
		super(CnnAttnLstmCRF, self).__init__()
		if model_config['random_embedding']:
			self.char_embeddings = nn.Embedding(data.char_alphabet_size, model_config['char_emb_dim'])
			self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(
				data.char_alphabet_size, model_config['char_emb_dim'])))
			self.char_drop = nn.Dropout(model_config['dropout'])
		else:
			char_emb_path = model_config['char_emb_file']
			self.pretrain_char_embedding, self.char_emb_dim = build_pretrain_embedding(char_emb_path, self.data.char_alphabet)
			self.char_embeddings.weight.data.copy_(torch.from_numpy(self.pretrain_char_embedding))
			self.char_drop = nn.Dropout(model_config['dropout'])

		self.word_embeddings = nn.Embedding(data.word_alphabet_size, model_config['word_emb_dim'])
		self.word_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(
			data.word_alphabet_size, model_config['word_emb_dim'])))

		self.intent_embeddings = nn.Embedding(data.intent_alphabet_size, model_config['intent_emb_dim'])
		self.intent_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(
			data.intent_alphabet_size, model_config['intent_emb_dim'])))

		self.lexi_embeddings = nn.Embedding(data.lexicon_alphabet_size, model_config['lexi_emb_dim'])
		self.lexi_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(
			data.lexicon_alphabet_size, model_config['lexi_emb_dim'])))

		self.word_drop = nn.Dropout(model_config['dropout'])

		self.char_cnn = nn.Conv1d(
			in_channels=model_config['char_emb_dim'], out_channels=model_config['cnn_hidden_dim'], kernel_size=3, padding=1)

		self.lstm = nn.LSTM(
			model_config['cnn_hidden_dim'] + model_config['intent_emb_dim'], model_config['lstm_hidden_dim']//2,
			num_layers=model_config['num_layers'], batch_first=model_config['batch_first'],
			bidirectional=model_config['bidirectional'])
		self.num_layers = model_config['num_layers']
		self.hidden_size = model_config['lstm_hidden_dim']//2
		self.drop_lstm = nn.Dropout(model_config['dropout'])

		self.hidden2tag = nn.Linear(model_config['lstm_hidden_dim'], data.label_alphabet_size + 2)
		self.crf = CRF(data.label_alphabet_size, model_config['gpu'])

		# multi-head-attention的分母，取值根号d_k
		temperature = np.power(model_config['char_emb_dim'], 0.5)
		self.attention = ScaledDotProductAttention(temperature)

	def forward(
			self, batch_char, batch_word, batch_intent, batch_lexicon, batch_char_len, mask, batch_lexicon_indices,
			batch_word_indices, batch_label=None):

		# char
		char_embeds = self.char_drop(self.char_embeddings(batch_char)).transpose(1, 2)
		char_cnn_out = self.char_cnn(char_embeds).transpose(1, 2)

		intent_embeds = self.intent_embeddings(batch_intent)
		char_intent_embeds = torch.repeat_interleave(intent_embeds, batch_char.size(1), dim=1)
		char_features = torch.cat([char_cnn_out, char_intent_embeds], 2)

		word_embeds = self.word_embeddings(batch_word)
		lexi_embeds = self.lexi_embeddings(batch_lexicon)
		batch_lexicon_indices, batch_word_indices = batch_lexicon_indices.unsqueeze(-1), batch_word_indices.unsqueeze(-1)
		# replace embedding
		word_features = word_embeds * batch_word_indices + lexi_embeds * batch_lexicon_indices

		# 第一个参数：源序列，第二个参数：目标序列
		attn_mask = get_attn_key_pad_mask(batch_word, batch_char)
		q = char_features  # (b, 32, 400)
		k = word_features  # (b, 32, 400)
		v = word_features  # (b, 32, 400)
		attn_output, _ = self.attention(q, k, v, attn_mask)

		# 由于固定padding长度，这里不采用动态RNN策略
		h0 = torch.zeros(self.num_layers*2, batch_char.size(0), self.hidden_size)
		c0 = torch.zeros(self.num_layers*2, batch_char.size(0), self.hidden_size)
		lstm_out, _ = self.lstm(attn_output, (h0, c0))

		# fc
		outputs = self.hidden2tag(lstm_out)

		# crf
		if batch_label is not None:
			total_loss = self.crf.neg_log_likelihood_loss(outputs, mask, batch_label)
			_, tag_seq = self.crf._viterbi_decode(outputs, mask)
			return total_loss, tag_seq
		else:
			_, tag_seq = self.crf._viterbi_decode(outputs, mask)
			return tag_seq

	@staticmethod
	def random_embedding(vocab_size, embedding_dim):
		pretrain_emb = np.empty([vocab_size, embedding_dim])
		scale = np.sqrt(3.0 / embedding_dim)
		for index in range(vocab_size):
			pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
		return pretrain_emb

	# 用lexicon_embedding替换word_embedding
	def repalce_embedding(self, word_embeds, lexi_embeds, batch_lexicon):
		new_word_embeddings = []
		for word_emb, lexi_emb, one_lexi in zip(word_embeds, lexi_embeds, batch_lexicon):
			word_emb_list = word_emb.tolist()
			lexi_emb_list = lexi_emb.tolist()
			one_lexi_list = one_lexi.tolist()
			for idx, lexi_id in enumerate(one_lexi_list):
				if lexi_id == 0:
					break
				elif lexi_id not in self.no_replaced_lexicon_id:
					word_emb_list[idx] = lexi_emb_list[idx]
			new_word_embeddings.append(word_emb_list)

		return new_word_embeddings

	# 计算需要替换的word位置的index
	def gen_word_index(self, batch_lexi):
		pass
