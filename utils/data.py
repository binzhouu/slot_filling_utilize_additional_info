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
import json
from constants import ROOT_PATH
from utils.trees import Trees
from utils.alphabet import Alphabet
from utils.functions import normalize_word

logger = logging.getLogger(__name__)
random.seed(34)


class Data(object):
	def __init__(self, data_config_file, alphabet_path, if_train=True):
		if if_train:
			with open(data_config_file, 'r') as rf:
				self.data_config = yaml.load(rf, Loader=yaml.FullLoader)
			# init data file
			mode = self.data_config['mode']
			self.data_file = os.path.join(ROOT_PATH, self.data_config['data'][mode])
			# init ac tree
			specific_words_file = os.path.join(ROOT_PATH, self.data_config['specific_words_file'])
			self.trees = Trees.build_trees(specific_words_file)
			# init alphabet
			self.char_alphabet = Alphabet('char')
			self.intent_alphabet = Alphabet('intent')
			self.label_alphabet = Alphabet('label', label=True)
			self.char_alphabet_size, self.intent_alphabet_size, self.label_alphabet_size = -1, -1, -1
			# pad length
			self.char_max_length = self.data_config['char_max_length']
			# read data file
			with open(self.data_file, 'r') as rf:
				self.corpus = rf.readlines()
			self.build_alphabet(alphabet_path)
			self.texts, self.ids = self.read_instance()
			self.train_texts, self.train_ids, self.dev_texts, self.dev_ids, self.test_texts, self.test_ids = self.sample_split()
		else:  # inference use
			self.char_alphabet = Alphabet('char', keep_growing=False)
			self.intent_alphabet = Alphabet('intent', keep_growing=False)
			self.label_alphabet = Alphabet('label', label=True, keep_growing=False)

	def build_alphabet(self, alphabet_path):
		for line in self.corpus:
			line = ast.literal_eval(line)
			char, char_label, seg_list, intent = line['char'], line['char_label'], line['word'], line['intent']
			for word in seg_list:
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
				# 抽象成一个字符
				self.char_alphabet.add(lexi_feat)
			# char
			for c in char:
				self.char_alphabet.add(normalize_word(c))
			# intent
			self.intent_alphabet.add(intent)
			# label
			for label in char_label:
				self.label_alphabet.add(label)
		# alphabet_size
		self.char_alphabet_size = self.char_alphabet.size()
		self.intent_alphabet_size = self.intent_alphabet.size()
		self.label_alphabet_size = self.label_alphabet.size()
		# close alphabet
		self.fix_alphabet()

		# write alphabet:
		if not os.path.exists(alphabet_path):
			with open(alphabet_path, 'wb') as wbf:
				pickle.dump(self.char_alphabet.instance2index, wbf)
				pickle.dump(self.intent_alphabet.instance2index, wbf)
				pickle.dump(self.label_alphabet.instance2index, wbf)
				pickle.dump(self.label_alphabet.instances, wbf)
				pickle.dump(self.char_alphabet_size, wbf)
				pickle.dump(self.intent_alphabet_size, wbf)
				pickle.dump(self.label_alphabet_size, wbf)

	def read_instance(self):
		"""
		这里读取完整读数据，不做截断，functions.py中做截断
		:return:
		"""
		texts, ids = [], []
		for idx, line in enumerate(self.corpus):
			line = ast.literal_eval(line)
			intent_id_list= []
			# word：'0010000' -> 合并成一个标签
			seq_char, seq_char_id_list, seq_label, seq_label_id_list = [], [], [], []
			char, char_label, seg_list, intent = line['char'], line['char_label'], line['word'], line['intent']
			# 存储one-hot形式的属性特征
			lexicons = []
			# 记录字符的index
			word_indices = []
			start = 0
			flag = True  # 判断跳至上一循环
			for word in seg_list:
				if flag is True:
					end = start + len(word)
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
					word_indices.append([start, end])

					# char
					# '0010000'
					if '1' in lexi_feat:
						seq_char.append(lexi_feat)
						seq_char_id_list.append(self.char_alphabet.get_index(lexi_feat))
						# ["B-room", "I-room", "I-room"]
						specific_word_label = char_label[start: end]
						tmp_label = [swl.split('-')[-1] for swl in specific_word_label]
						if len(set(tmp_label)) > 1:
							# 判断是否过滤该条数据
							# print('Be filtered: %s' % line['text'], word, tmp_label)
							flag = False
						else:
							assert len(set(tmp_label)) == 1
							if tmp_label[0] == 'O':
								tmp_label = 'O'
							else:
								tmp_label = 'B' + '-' + tmp_label[0]
							seq_label += [tmp_label]
							seq_label_id_list += [self.label_alphabet.get_index(tmp_label)]
					# '0000000'
					else:
						for c in word:
							seq_char.append(c)
							seq_char_id_list.append(self.char_alphabet.get_index(normalize_word(c)))
						seq_label += char_label[start: end]
						seq_label_id_list += [self.label_alphabet.get_index(cl) for cl in char_label[start: end]]

					start = end
				else:
					break  # 跳至下一个corpus

			intent_id_list.append(self.intent_alphabet.get_index(intent))

			if idx % 10000 == 0:
				logger.info('read instance : %s' % idx)

			if flag is True:
				# text, char, intent, sequence_label
				texts.append([line['text'], seq_char, intent, seq_label])
				ids.append([seq_char_id_list, intent_id_list, seq_label_id_list])

		# 新形式的corpus的保存下来,方便查bug
		output_path = self.data_config['data']['output']
		with open(output_path, 'w') as wf:
			for text in texts:
				line_data = dict()
				line_data['text'] = text[0]
				line_data['char'] = text[1]
				line_data['intent'] = text[2]
				line_data['char_label'] = text[-1]
				wf.write(json.dumps(line_data, ensure_ascii=False) + '\n')

		return texts, ids

	def fix_alphabet(self):
		self.char_alphabet.close()
		self.intent_alphabet.close()
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
