# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-06-06
Description: 
"""
import os
import logging
import pickle
import yaml
import random
import ast
from constants import ROOT_PATH
from utils.trees import Trees
from utils.alphabet import Alphabet
from utils.functions import normalize_word

logger = logging.getLogger(__name__)
random.seed(34)


class Data(object):
	def __init__(self, data_config_file, alphabet_path, if_train=True):
		with open(data_config_file, 'r') as rf:
			self.data_config = yaml.load(rf, Loader=yaml.FullLoader)
		# init data file
		mode = self.data_config['mode']
		self.data_file = os.path.join(ROOT_PATH, self.data_config['data'][mode])
		specific_words_file = os.path.join(ROOT_PATH, self.data_config['specific_words_file'])
		# init ac tree
		self.trees = Trees.build_trees(specific_words_file)
		# init alphabet
		self.char_alphabet = Alphabet('char')
		self.word_alphabet = Alphabet('word')
		self.intent_alphabet = Alphabet('intent')
		self.lexicon_alphabet = Alphabet('lexicon')
		self.label_alphabet = Alphabet('label', label=True)
		self.char_alphabet_size, self.word_alphabet_size, self.intent_alphabet_size, self.lexicon_alphabet_size, \
			self.label_alphabet_size = -1, -1, -1, -1, -1

		self.char_max_length, self.word_max_length = self.data_config['char_max_length'], self.data_config['word_max_length']

		if if_train:
			with open(self.data_file, 'r') as rf:
				self.corpus = rf.readlines()
			self.build_alphabet(alphabet_path)
			self.texts, self.ids = self.read_instance()
			self.train_texts, self.train_ids, self.dev_texts, self.dev_ids, self.test_texts, self.test_ids = self.sample_split()

	def build_alphabet(self, alphabet_path):
		for line in self.corpus:
			line = ast.literal_eval(line)
			char, char_label, seg_list, intent = line['char'], line['char_label'], line['word'], line['intent']
			for word in seg_list:
				# word
				self.word_alphabet.add(normalize_word(word))
				# lexicon
				lexi_feat = []
				for lexi_type, lb in self.trees.lexi_trees.items():
					lexi_feat.append(lb.search(word))
				for n in range(len(lexi_feat)):
					if lexi_feat[n] is None or lexi_feat[n] == '_STEM_':
						lexi_feat[n] = 0
					else:
						lexi_feat[n] = 1
				lexi_feat = ''.join([str(i) for i in lexi_feat])
				self.lexicon_alphabet.add(lexi_feat)
				# char
				for char in word:
					self.char_alphabet.add(normalize_word(char))
			# intent
			self.intent_alphabet.add(intent)
			# label
			for label in char_label:
				self.label_alphabet.add(label)
		# alphabet_size
		self.char_alphabet_size = self.char_alphabet.size()
		self.word_alphabet_size = self.word_alphabet.size()
		self.intent_alphabet_size = self.intent_alphabet.size()
		self.lexicon_alphabet_size = self.lexicon_alphabet.size()
		self.label_alphabet_size = self.label_alphabet.size()
		# close alphabet
		self.fix_alphabet()

		# write alphabet:
		if not os.path.exists(alphabet_path):
			with open(alphabet_path, 'wb') as wbf:
				pickle.dump(self.char_alphabet.instance2index, wbf)
				pickle.dump(self.word_alphabet.instance2index, wbf)
				pickle.dump(self.intent_alphabet.instance2index, wbf)
				pickle.dump(self.lexicon_alphabet.instance2index, wbf)
				pickle.dump(self.label_alphabet.instance2index, wbf)
				pickle.dump(self.char_alphabet_size, wbf)
				pickle.dump(self.word_alphabet_size, wbf)
				pickle.dump(self.intent_alphabet_size, wbf)
				pickle.dump(self.lexicon_alphabet_size, wbf)
				pickle.dump(self.label_alphabet_size, wbf)

	def read_instance(self):
		"""
		这里读取完整读数据，不做截断，functions.py中做截断
		:return:
		"""
		texts, ids = [], []
		for line in self.corpus:
			line = ast.literal_eval(line)
			char_id_list, word_id_list, intent_id_list, lexicon_id_list, label_id_list = [], [], [], [], []
			char, char_label, seg_list, intent = line['char'], line['char_label'], line['word'], line['intent']
			# 存储one-hot形式的属性特征
			lexicons = []
			for word in seg_list:
				word_id = self.word_alphabet.get_index(normalize_word(word))
				word_id_list.append(word_id)

				lexi_feat = []
				for lexi_type, lb in self.trees.lexi_trees.items():
					lexi_feat.append(lb.search(word))
				for n in range(len(lexi_feat)):
					if lexi_feat[n] is None or lexi_feat[n] == '_STEM_':
						lexi_feat[n] = 0
					else:
						lexi_feat[n] = 1
				lexi_feat = ''.join([str(i) for i in lexi_feat])
				lexicons.append(lexi_feat)
				lexicon_id_list.append(self.lexicon_alphabet.get_index(lexi_feat))

			for c in char:
				char_id_list.append(self.char_alphabet.get_index(normalize_word(c)))

			intent_id_list.append(self.intent_alphabet.get_index(intent))

			for label in char_label:
				label_id_list.append(self.label_alphabet.get_index(label))

			# char, word, intent, lexicon_feat, sequence_label
			texts.append([char, seg_list, [intent], lexicons, char_label])
			ids.append([char_id_list, word_id_list, intent_id_list, lexicon_id_list, label_id_list])

		char_length_list = sorted([len(text[0]) for text in texts], reverse=True)
		logger.info("top 10 max length in chars: %s" % char_length_list[:10])

		return texts, ids

	def fix_alphabet(self):
		self.char_alphabet.close()
		self.word_alphabet.close()
		self.intent_alphabet.close()
		self.lexicon_alphabet.close()
		self.label_alphabet.close()

	# data sampling
	def sample_split(self):
		sampling_rate = self.data_config['sampling_rate']
		indexes = list(range(len(self.ids)))
		random.shuffle(indexes)
		shuffled_texts = [self.texts[i] for i in indexes]
		shuffled_ids = [self.ids[i] for i in indexes]
		logger.info('Top 10 shuffled indexes: %s' % indexes[:10])

		n = int(len(shuffled_ids) * sampling_rate)
		dev_texts, dev_ids = shuffled_texts[:n], shuffled_ids[:n]
		test_texts, test_ids = shuffled_texts[n:2*n], shuffled_ids[n:2*n]
		train_texts, train_ids = shuffled_texts[2*n:], shuffled_ids[2*n:]

		return train_texts, train_ids, dev_texts, dev_ids, test_texts, test_ids


if __name__ == '__main__':
	data_config_file = os.path.join(ROOT_PATH, 'conf/data_config.yaml')
	alphabet_path = os.path.join(ROOT_PATH, 'saved_models/cnn_attn_lstm_crf/data/alphabet.dset')
	data = Data(data_config_file, alphabet_path)
	pass
