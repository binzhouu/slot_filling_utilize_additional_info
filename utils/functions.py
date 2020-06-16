# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-06-08
Description: 
"""
import numpy as np
import pickle
import torch
import logging
import copy

logger = logging.getLogger(__name__)


def normalize_word(word):
	new_word = ""
	for char in word:
		if char.isdigit():  # 如果字符串为数字组成，则为True
			# print('char:', char)
			new_word += '0'
		# print('new_word:', new_word)
		else:
			new_word += char
	return new_word


def batch_char_sequence_labeling_process(
		input_batch_list: list, gpu: bool, char_max_length: int, word_max_length: int, no_replaced_id: list,
		if_train: bool
) -> tuple:
	"""

	:param input_batch_list: [[char],[word],[intent],[lexicon],[label]]
	:param gpu:
	:param char_max_length: max length of char sequence
	:param word_max_length: max length of word sequence
	:param no_replaced_id: no_replaced_lexicon_id
	:param if_train: train process or inference process
	:return:
	"""
	batch_size = len(input_batch_list)
	# char和word级别的特征，按照max_length的长度做截断
	chars = [sent[0][:char_max_length] for sent in input_batch_list]
	words = [sent[1][:word_max_length] for sent in input_batch_list]
	intents = [sent[2] for sent in input_batch_list]
	lexicons = [sent[3][:word_max_length] for sent in input_batch_list]
	# 计算replaced index
	lexicon_indices = copy.deepcopy(lexicons)
	for lexicon in lexicon_indices:
		for n in range(len(lexicon)):
			if lexicon[n] in no_replaced_id:
				lexicon[n] = 0
			else:
				lexicon[n] = 1

	# train use
	if if_train:
		labels = [sent[-1][:char_max_length] for sent in input_batch_list]
		# 字符
		char_seq_lengths = torch.tensor(list(map(len, chars)), dtype=torch.long)
		char_seq_tensor = torch.zeros((batch_size, char_max_length), requires_grad=if_train).long()
		label_seq_tensor = torch.zeros((batch_size, char_max_length), requires_grad=if_train).long()
		mask = torch.zeros((batch_size, char_max_length), requires_grad=if_train).byte()

		# padding
		for idx, (seq, label, seq_len) in enumerate(zip(chars, labels, char_seq_lengths)):
			length = seq_len.item()
			char_seq_tensor[idx, :length] = torch.tensor(seq, dtype=torch.long)
			label_seq_tensor[idx, :length] = torch.tensor(label, dtype=torch.long)
			mask[idx, :length] = torch.tensor([1] * length, dtype=torch.long)

		# intent
		intent_seq_tensor = torch.tensor(intents).reshape(batch_size, -1)

		# rank
		char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
		char_seq_tensor = char_seq_tensor[char_perm_idx]
		intent_seq_tensor = intent_seq_tensor[char_perm_idx]
		label_seq_tensor = label_seq_tensor[char_perm_idx]
		mask = mask[char_perm_idx]

		# 词
		word_seq_lengths = torch.tensor(list(map(len, words)), dtype=torch.long)
		word_seq_tensor = torch.zeros((batch_size, word_max_length), requires_grad=if_train).long()
		lexicon_seq_tensor = torch.zeros((batch_size, word_max_length), requires_grad=if_train).long()
		# lexicon index
		# padding的位置需要置为0（模型部分要做乘法）
		lexicon_indices_seq_tensor = torch.zeros((batch_size, word_max_length), requires_grad=False).float()
		# word index
		# padding的位置需要置为1
		word_indices_seq_tensor = torch.ones((batch_size, word_max_length), requires_grad=False).float()

		# assignment
		for idx, (seq, lexi, lexi_index, seq_len) in enumerate(zip(words, lexicons, lexicon_indices, word_seq_lengths)):
			length = seq_len.item()
			word_seq_tensor[idx, :length] = torch.tensor(seq, dtype=torch.long)
			lexicon_seq_tensor[idx, :length] = torch.tensor(lexi, dtype=torch.long)
			lexicon_indices_seq_tensor[idx, :length] = torch.tensor(lexi_index, dtype=torch.long)
			word_indices_seq_tensor[idx, :length] = \
				torch.tensor(1, dtype=torch.long) - torch.tensor(lexi_index, dtype=torch.long)

		# rank
		word_seq_tensor = word_seq_tensor[char_perm_idx]
		lexicon_seq_tensor = lexicon_seq_tensor[char_perm_idx]
		word_seq_lengths = word_seq_lengths[char_perm_idx]
		lexicon_indices_seq_tensor = lexicon_indices_seq_tensor[char_perm_idx]
		word_indices_seq_tensor = word_indices_seq_tensor[char_perm_idx]

	# inference use
	else:
		char_seq_lengths = torch.tensor(list(map(len, chars)), dtype=torch.long)
		char_seq_tensor = torch.zeros((batch_size, char_max_length), requires_grad=if_train).long()
		mask = torch.zeros((batch_size, char_max_length), requires_grad=if_train).long().byte()

		# padding
		for idx, (seq, seq_len) in enumerate(zip(chars, char_seq_lengths)):
			length = seq_len.item()
			char_seq_tensor[idx, :length] = torch.tensor(seq, dtype=torch.long)
			mask[idx, :length] = torch.tensor([1] * length, dtype=torch.long)

		# intent
		intent_seq_tensor = torch.tensor(intents).reshape(batch_size, -1)

		# rank
		char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
		char_seq_tensor = char_seq_tensor[char_perm_idx]
		intent_seq_tensor = intent_seq_tensor[char_perm_idx]
		mask = mask[char_perm_idx]

		# 词
		word_seq_lengths = torch.tensor(list(map(len, words)), dtype=torch.long)
		word_seq_tensor = torch.zeros((batch_size, word_max_length), requires_grad=if_train).long()
		lexicon_seq_tensor = torch.zeros((batch_size, word_max_length), requires_grad=if_train).long()
		# lexicon index
		# padding的位置需要置为0（forward部分要做乘法）
		lexicon_indices_seq_tensor = torch.zeros((batch_size, word_max_length), requires_grad=False).float()
		# word index
		# padding的位置需要置为1
		word_indices_seq_tensor = torch.ones((batch_size, word_max_length), requires_grad=False).float()

		# assignment
		for idx, (seq, lexi, lexi_index, seq_len) in enumerate(zip(words, lexicons, lexicon_indices, word_seq_lengths)):
			length = seq_len.item()
			word_seq_tensor[idx, :length] = torch.tensor(seq, dtype=torch.long)
			lexicon_seq_tensor[idx, :length] = torch.tensor(lexi, dtype=torch.long)
			lexicon_indices_seq_tensor[idx:, length] = torch.tensor(lexi_index, dtype=torch.long)
			word_indices_seq_tensor[idx: length] = \
				torch.tensor(1, dtype=torch.long) - torch.tensor(lexi_index, dtype=torch.long)

		# rank
		word_seq_tensor = word_seq_tensor[char_perm_idx]
		lexicon_seq_tensor = lexicon_seq_tensor[char_perm_idx]
		word_seq_lengths = word_seq_lengths[char_perm_idx]
		lexicon_indices_seq_tensor = lexicon_indices_seq_tensor[char_perm_idx]
		word_indices_seq_tensor = word_indices_seq_tensor[char_perm_idx]

		#
		label_seq_tensor = None

	# recover:
	_, char_seq_recover = char_perm_idx.sort(0, descending=False)
	_, word_seq_recover = char_perm_idx.sort(0, descending=False)

	if gpu:
		char_seq_tensor = char_seq_tensor.cuda()
		word_seq_tensor = word_seq_tensor.cuda()
		intent_seq_tensor = intent_seq_tensor.cuda()
		lexicon_seq_tensor = lexicon_seq_tensor.cuda()
		label_seq_tensor = label_seq_tensor.cuda()
		char_seq_lengths = char_seq_lengths.cuda()
		word_seq_lengths = word_seq_lengths.cuda()
		char_seq_recover = char_seq_recover.cuda()
		word_seq_recover = word_seq_recover.cuda()
		mask = mask.cuda()
		lexicon_indices_seq_tensor = lexicon_indices_seq_tensor.cuda()
		word_indices_seq_tensor = word_indices_seq_tensor.cuda()

	return char_seq_tensor, word_seq_tensor, intent_seq_tensor, lexicon_seq_tensor, label_seq_tensor, char_seq_lengths, \
		mask, lexicon_indices_seq_tensor, word_indices_seq_tensor


def build_pretrain_embedding(embedding_path, word_alphabet, norm=True):
	# embedd_dict = dict()
	embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
	alphabet_size = len(word_alphabet.instance2index) + 1
	scale = np.sqrt(3.0 / embedd_dim)
	pretrain_emb = np.nan_to_num(np.empty([alphabet_size, embedd_dim]))
	perfect_match = 0
	case_match = 0
	not_match = 0
	for word, index in word_alphabet.iteritems():
		if word in embedd_dict:
			if norm:
				pretrain_emb[index, :] = norm2one(embedd_dict[word])
			else:
				pretrain_emb[index, :] = embedd_dict[word]
			perfect_match += 1
		elif word.lower() in embedd_dict:
			if norm:
				pretrain_emb[index, :] = norm2one(embedd_dict[word.lower()])
			else:
				pretrain_emb[index, :] = embedd_dict[word.lower()]
			case_match += 1
		else:
			pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
			not_match += 1
	pretrained_size = len(embedd_dict)
	print("Embedding:\n     pretrain word:%s, perfect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
		pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / alphabet_size))
	return pretrain_emb, embedd_dim


def load_pretrain_emb(embedding_path):
	# embedd_dim = -1
	with open(embedding_path, 'rb') as rbf:
		embedd_dict = pickle.load(rbf)
		embedd_dim = embedd_dict[list(embedd_dict.keys())[0]].size
	return embedd_dict, embedd_dim


def norm2one(vec):
	root_sum_square = np.sqrt(np.sum(np.square(vec)))
	return vec / root_sum_square


# 计算sel-attn中的需要传入的mask
def get_attn_key_pad_mask(seq_k, seq_q):
	""" For masking out the padding part of key sequence. """

	# Expand to fit the shape of key query attention matrix.
	len_q = seq_q.size(1)
	padding_mask = seq_k.eq(0)  # padding的部分置为1
	padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # batch x len_seq_q x len_seq_k

	return padding_mask


def predict_check(pred_variable, gold_variable, mask_variable):
	pred = pred_variable.cpu().data.numpy()
	gold = gold_variable.cpu().data.numpy()
	mask = mask_variable.cpu().data.numpy()
	overlap = (pred == gold)
	right_token = np.sum(overlap * mask)
	total_token = mask.sum()
	return right_token, total_token


def evaluate_model(data, model, name, config, alphabet, encoder_type):
	assert name in ['dev', 'test']
	if name == 'dev':
		instances = data.dev_ids
	elif name == 'test':
		instances = data.test_ids
	else:
		instances = []

	pred_scores = []
	pred_results = []
	gold_results = []

	model.eval()
	batch_size = config['batch_size']
	train_num = len(instances)

	total_batch = train_num // batch_size + 1
	total_loss = 0
	for batch_id in range(total_batch):
		start = batch_id * batch_size
		end = (batch_id + 1) * batch_size
		if end > train_num:
			end = train_num
		instance = instances[start:end]
		if not instance:
			continue
		if encoder_type == 'cnn_attn_lstm_crf':
			batch_char, batch_word, batch_intent, batch_lexicon, batch_label, batch_char_len, mask, batch_lexicon_indices, \
				batch_word_indices = batch_char_sequence_labeling_process(
					instance, config['gpu'], config['char_max_length'], config['word_max_length'],
					config['no_replaced_lexicon_id'], True)
			with torch.no_grad():  # 防止在验证阶段造成梯度累计
				loss, tag_seq = model(
					batch_char, batch_word, batch_intent, batch_lexicon, batch_char_len, mask, batch_lexicon_indices,
					batch_word_indices, batch_label)
				loss = loss.item()
		else:
			raise ValueError('No Model')
		# id to label
		pred_label, gold_label = recover_label(tag_seq, batch_label, mask, alphabet)
		pred_results += pred_label
		gold_results += gold_label
		total_loss += loss
	logger.info('%s pre_results: %s' % (name, len(pred_results)))
	logger.info('%s gold_results: %s' % (name, len(gold_results)))
	acc, p, r, f = get_ner_fmeasure(gold_results, pred_results)

	return total_loss, acc, p, r, f, pred_results, pred_scores


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, sentence_classification=False):
	batch_size = gold_variable.size(0)
	if sentence_classification:
		pred_tag = pred_variable.cpu().data.numpy().tolist()
		gold_tag = gold_variable.cpu().data.numpy().tolist()
		pred_label = [label_alphabet.get_instance(pred) for pred in pred_tag]
		gold_label = [label_alphabet.get_instance(gold) for gold in gold_tag]
	else:
		seq_len = gold_variable.size(1)
		mask = mask_variable.cpu().data.numpy()
		pred_tag = pred_variable.cpu().data.numpy()
		gold_tag = gold_variable.cpu().data.numpy()
		pred_label = []
		gold_label = []
		for idx in range(batch_size):
			pred = \
				[label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
			gold = \
				[label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
			assert (len(pred) == len(gold))
			pred_label.append(pred)
			gold_label.append(gold)
	return pred_label, gold_label


def get_ner_fmeasure(golden_lists, predict_lists, label_type='BIO'):
	sent_num = len(golden_lists)  # 句子的数量
	golden_full = []
	predict_full = []
	right_full = []
	right_tag = 0
	all_tag = 0
	for idx in range(sent_num):
		golden_list = golden_lists[idx]
		predict_list = predict_lists[idx]
		for idy in range(len(golden_list)):
			if golden_list[idy] == predict_list[idy]:
				right_tag += 1
		all_tag += len(golden_list)
		if label_type == "BMES" or label_type == 'BIOES':
			gold_matrix = get_ner_BMES(golden_list)
			pred_matrix = get_ner_BMES(predict_list)
		else:
			gold_matrix = get_ner_BIO(golden_list)
			pred_matrix = get_ner_BIO(predict_list)
		right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
		golden_full += gold_matrix
		predict_full += pred_matrix
		right_full += right_ner
	right_num = len(right_full)
	golden_num = len(golden_full)
	predict_num = len(predict_full)
	logger.info("label_type: %s, jiaoji: %s, true: %s, predict: %s" % (label_type, right_num, golden_num, predict_num))
	if predict_num == 0:
		precision = -1
	else:
		precision = (right_num + 0.0) / predict_num
	if golden_num == 0:
		recall = -1
	else:
		recall = (right_num + 0.0) / golden_num
	if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
		f_measure = -1
	else:
		f_measure = 2 * precision * recall / (precision + recall)

	accuracy = (right_tag + 0.0) / all_tag
	if label_type.upper().startswith("B-"):
		logger.info("golden_num= %s, predict_num= %s, right_num= %s" % (golden_num, predict_num, right_num))
	else:
		logger.info("Right token= %s, All token= %s, acc= %s" % (right_tag, all_tag, accuracy))
	return accuracy, precision, recall, f_measure


def get_ner_BIO(label_list):
	list_len = len(label_list)
	begin_label = 'B-'
	inside_label = 'I-'
	whole_tag = ''
	index_tag = ''
	tag_list = []
	stand_matrix = []
	for i in range(0, list_len):
		current_label = label_list[i].upper()
		if begin_label in current_label:
			if index_tag == '':
				whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
				index_tag = current_label.replace(begin_label, "", 1)
			else:
				tag_list.append(whole_tag + ',' + str(i - 1))
				whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
				index_tag = current_label.replace(begin_label, "", 1)
		elif inside_label in current_label:
			if current_label.replace(inside_label, "", 1) == index_tag:
				whole_tag = whole_tag
			else:
				if (whole_tag != '') & (index_tag != ''):
					tag_list.append(whole_tag + ',' + str(i - 1))
				whole_tag = ''
				index_tag = ''
		else:
			if (whole_tag != '') & (index_tag != ''):
				tag_list.append(whole_tag + ',' + str(i - 1))
			whole_tag = ''
			index_tag = ''
	if (whole_tag != '') & (index_tag != ''):
		tag_list.append(whole_tag)
	tag_list_len = len(tag_list)

	for i in range(0, tag_list_len):
		if len(tag_list[i]) > 0:
			tag_list[i] = tag_list[i] + ']'
			insert_list = reverse_style(tag_list[i])
			stand_matrix.append(insert_list)
	return stand_matrix


def get_ner_BMES(label_list):
	# list_len = len(word_list)
	# assert(list_len == len(label_list)), "word list size unmatch with label list"
	list_len = len(label_list)
	begin_label = 'B-'
	end_label = 'E-'
	single_label = 'S-'
	whole_tag = ''
	index_tag = ''
	tag_list = []
	stand_matrix = []
	for i in range(0, list_len):
		# wordlabel = word_list[i]
		current_label = label_list[i].upper()
		if begin_label in current_label:
			if index_tag != '':
				tag_list.append(whole_tag + ',' + str(i - 1))
			whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
			index_tag = current_label.replace(begin_label, "", 1)

		elif single_label in current_label:
			if index_tag != '':
				tag_list.append(whole_tag + ',' + str(i - 1))
			whole_tag = current_label.replace(single_label, "", 1) + '[' + str(i)
			tag_list.append(whole_tag)
			whole_tag = ""
			index_tag = ""
		elif end_label in current_label:
			if index_tag != '':
				tag_list.append(whole_tag + ',' + str(i))
			whole_tag = ''
			index_tag = ''
		else:
			continue
	if (whole_tag != '') & (index_tag != ''):
		tag_list.append(whole_tag)
	tag_list_len = len(tag_list)

	for i in range(0, tag_list_len):
		if len(tag_list[i]) > 0:
			tag_list[i] = tag_list[i] + ']'
			insert_list = reverse_style(tag_list[i])
			stand_matrix.append(insert_list)
	# print stand_matrix
	return stand_matrix


def reverse_style(input_string):
	target_position = input_string.index('[')
	input_len = len(input_string)
	output_string = input_string[target_position:input_len] + input_string[0:target_position]
	return output_string
