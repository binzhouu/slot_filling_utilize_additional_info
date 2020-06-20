# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-03-24
"""
import os
import re
from collections import defaultdict
from constants import ROOT_PATH
from utils.lexicon import LexiconBuilder


# 构建字典树
class Trees(object):
	def __init__(self, word_dict, lexi_trees):
		self.word_dict = word_dict
		self.lexi_trees = lexi_trees  # {'lexi_type':LexiconBuilder}

	@classmethod
	def build_trees(cls, specific_words_file):
		lexi_trees = {}
		word_dict = defaultdict(list)
		with open(specific_words_file, 'r') as rf:
			for line in rf:
				line = re.split(r'[\s\t]+', line.strip())
				token, value = line[0], line[1]
				if token == 'word':
					continue
				else:
					word_dict[value].append(token)
		# word_dict需要预先排序，保证推理时获得的特征顺序一致
		for lexi_type, tokens in sorted(word_dict.items(), key=lambda x:x[0], reverse=False):
			lb = LexiconBuilder(tokens, lexi_type)
			lexi_trees[lexi_type] = lb
		return cls(word_dict, lexi_trees)


if __name__ == '__main__':
	pass
