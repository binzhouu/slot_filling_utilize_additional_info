# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-06-06
Description: 
"""
from datetime import datetime
import os
import pickle
import torch
import numpy as np
import re
import json
import yaml
import grpc
from constants import ROOT_PATH
from utils.data import Data
from utils.trees import Trees
from model import BilstmCrf
from utils.functions import normalize_word, batch_char_sequence_labeling_process
from protos.nlp_basic_pb2 import Request
from protos import nlp_basic_pb2_grpc

model_config_file = os.path.join(ROOT_PATH, 'conf/model_config.yaml')
inference_config_file = os.path.join(ROOT_PATH, 'conf/inference_config.yaml')
data_config_file = os.path.join(ROOT_PATH, 'conf/data_config.yaml')


class SlotModel(object):
	def __init__(self, ip_port):
		"""

		:param ip_port: ai-nlp-basic ip:port
		"""
		configs = self.read_configs(1)
		dset_path, model_path = configs['alphabet_path'], configs['model_path']
		self.data = Data('', '', False)
		# 推理阶段需要构建ac tree
		self.data.trees = Trees.build_trees(configs['specific_words_file'])
		with open(dset_path, 'rb') as rbf:
			self.data.char_alphabet.instance2index = pickle.load(rbf)  # keep_growing: False
			self.data.intent_alphabet.instance2index = pickle.load(rbf)
			self.data.label_alphabet.instance2index = pickle.load(rbf)
			self.data.label_alphabet.instances = pickle.load(rbf)
			self.data.char_alphabet_size = pickle.load(rbf)
			self.data.intent_alphabet_size = pickle.load(rbf)
			self.data.label_alphabet_size = pickle.load(rbf)

		self.model = BilstmCrf(self.data, configs)
		self.model.load_state_dict(torch.load(model_path, map_location=configs['map_location']))
		self.model.eval()
		self.model.to(configs['device'])

		self.gpu = configs['gpu']
		self.char_max_length = configs['char_max_length']

		self.channel = grpc.insecure_channel(ip_port)
		self.stub = nlp_basic_pb2_grpc.NLPBasicServerStub(self.channel)

	def inference(self, text, intent, session_keep, previous_intent, trace_id=''):
		"""

		:param text:
		:param intent: 当前意图
		:param session_keep:
		:param previous_intent: 上一轮意图
		:param trace_id:
		:return:
		"""
		# 如果存在多轮对话，且当前intent为空，取上一轮text的意图
		if session_keep and intent is None:
			intent = previous_intent
		if intent is None:  # label_alphabet中的None是str类型
			intent = 'None'
		instance, instance_ids = [], []
		# 处理当前会话
		new_char, seq_char, seq_char_id_list, seq_label, seq_label_id_list = [], [], [], [], []
		char, seg_list = list(text), self.process(self.stub, text, trace_id)
		# 存储one-hot形式的属性特征
		lexicons = []
		# word level
		# 记录字符的index
		word_indices = []
		start = 0
		for word in seg_list:
			end = start + len(word)
			lexi_feat = []
			for lexi_type, lb in self.data.trees.lexi_trees.items():
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
				seq_char_id_list.append(self.data.char_alphabet.get_index(lexi_feat))
				new_char.append(''.join(char[start: end]))
			else:  # '0000000'
				for c in word:
					seq_char.append(c)
					seq_char_id_list.append(self.data.char_alphabet.get_index(normalize_word(c)))
					new_char.append(c)
			start = end
		# intent
		intent_id = self.data.intent_alphabet.get_index(intent)
		instance.append([seq_char, [intent]])
		instance_ids.append([seq_char_id_list, [intent_id]])
		# instance process
		batch_char, batch_intent, batch_char_len, mask, batch_char_recover, _ = \
			batch_char_sequence_labeling_process(instance_ids, self.gpu, self.char_max_length, False)
		tag_seq = self.model(batch_char, batch_intent, batch_char_len, mask)
		# label recover
		pred_result = self.predict_recover_label(tag_seq, mask, self.data.label_alphabet)
		pred_result = list(np.array(pred_result).reshape(len(seq_char), ))
		result = self.slot_concat(new_char, pred_result)

		return result

	@staticmethod
	def process(stub, text, trace_id):
		"""
		grpc server tokenize
		:param stub: grpc stub
		:param text:
		:param trace_id:
		:return:
		"""
		type = 'tokenize'
		request = Request(trace_id=trace_id, text=text, type=type)
		response = stub.process(request)
		result = response.result
		pairs = json.loads(result)['tokens']
		seg_list = [pair[0] for pair in pairs]

		return seg_list

	def close(self):
		self.channel.close()

	@staticmethod
	def read_configs(model_num: int) -> dict:
		"""

		:param model_num: specify a certain model
		0 means cnn_attn_lstm_crf
		1 means bilstm_crf
		:return:
		"""
		with open(model_config_file, 'r') as rf:
			configs = yaml.load(rf, Loader=yaml.FullLoader)
		# 读取设备基本属性
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		gpu = True if device.type == 'cuda' else False
		map_location = 'cpu' if gpu is False else None
		configs.update({'device': device, 'gpu': gpu, 'map_location': map_location})
		# 读取model_num对应的模型超参数
		for k, v in configs['model'][model_num].items():
			configs[k] = v
		del configs['model']
		# data config
		with open(data_config_file, 'r') as rf:
			data_config = yaml.load(rf, Loader=yaml.FullLoader)
		configs.update({
			'char_max_length': data_config['char_max_length'], 'specific_words_file': data_config['specific_words_file']})
		# inference_config
		with open(inference_config_file, 'r') as rf:
			inference_config = yaml.load(rf, Loader=yaml.FullLoader)
		model_path_head = '/'.join(configs['model_path'].split('/')[:-1])
		# update model path
		configs['model_path'] = os.path.join(ROOT_PATH, model_path_head, inference_config['model_name'])
		configs['alphabet_path'] = os.path.join(ROOT_PATH, configs['alphabet_path'], inference_config['alphabet_name'])
		configs['user_dict'] = os.path.join(ROOT_PATH, inference_config['user_dict'])

		return configs

	@staticmethod
	def predict_recover_label(pred_variable, mask_variable, label_alphabet):
		"""
		recover label id to name
		:param pred_variable:
		:param mask_variable:
		:param label_alphabet:
		:return:
		"""
		seq_len = pred_variable.size(1)
		mask = mask_variable.cpu().data.numpy()
		pred_tag = pred_variable.cpu().data.numpy()
		pred_label = []
		pred = [label_alphabet.get_instance(pred_tag[0][idy]) for idy in range(seq_len) if mask[0][idy] != 0]
		pred_label.append(pred)
		return pred_label

	@staticmethod
	def slot_concat(text_list, pred_result):
		pred_index = []
		start = 0
		for n in range(len(pred_result)):
			end = start + len(text_list[n])
			pred_index.append((start, end))
			start = end
		index_tmp, start, end = [], 0, 0
		entity_tmp = list()
		x = text_list
		y = pred_result
		entity = ''
		for n in range(len(y)):
			max_n = len(y) - 1
			if n < max_n:
				if y[n][0] == 'B' and len(entity) == 0:
					entity = y[n][2:] + ':' + x[n]
					start, end = pred_index[n][0], pred_index[n][-1]
				elif y[n][0] == 'I' and len(entity) != 0:
					entity += x[n]
					end = pred_index[n][-1]
				elif y[n][0] == 'O' and len(entity) != 0:
					entity_tmp.append(entity)
					index_tmp.append((start, end))
					entity = ''
				elif y[n][0] == 'B' and len(entity) != 0:  # B 连着 B 的情况
					entity_tmp.append(entity)
					index_tmp.append((start, end))
					entity = y[n][2:] + ':' + x[n]  # 重置entity
					start, end = pred_index[n][0], pred_index[n][-1]
			else:  # n == max_n
				if y[n][0] == 'B' and len(entity) == 0:
					entity = y[n][2:] + ':' + x[n]
					entity_tmp.append(entity)
					start, end = pred_index[n][0], pred_index[n][-1]
					index_tmp.append((start, end))
					entity = ''
				elif y[n][0] == 'B' and len(entity) != 0:  # B 连着 B 的情况
					entity_tmp.append(entity)
					index_tmp.append((start, end))
					entity = y[n][2:] + ':' + x[n]  # 重置entity
					start, end = pred_index[n][0], pred_index[n][-1]
					entity_tmp.append(entity)
					index_tmp.append((start, end))
				elif y[n][0] == 'I' and len(entity) != 0:
					entity += x[n]
					entity_tmp.append(entity)
					end = pred_index[n][-1]
					index_tmp.append((start, end))
				elif y[n][0] == 'O' and len(entity) != 0:
					entity_tmp.append(entity)
					index_tmp.append((start, end))
		result = []
		result_text = []
		for n in range(len(entity_tmp)):
			entity_list = []
			ent, idx_pair = entity_tmp[n], index_tmp[n]
			slot_label = re.split(r':', ent)[0]
			entity_list.append(slot_label)

			slot_name = re.split(r':', ent)[1]
			entity_list.append(slot_name)

			entity_list.append(idx_pair)
			result_text.append(entity_list)
		result.append(result_text)
		result = result[0] if result else result
		return result


if __name__ == '__main__':
	texts = [
		'开启电扇的消毒电子产品模式', '开卧房UV杀菌功能',
		'将客厅哈哈的色温调为100', '将客厅吊扇的色温调为100', '将客厅灯的色温调为100', '将客厅小黑的色温调为100', '将客厅小红的色温调为100']
	intents = [
		'open_function', 'open_function',
		'set_attribute', 'set_attribute', 'set_attribute', 'set_attribute', 'set_attribute']
	slot_model = SlotModel('172.16.246.53:31644')
	for t, it in zip(texts, intents):
		start_time = datetime.now()
		res = slot_model.inference(t, it, False, None)
		print('text: %s, intent: %s res: %s, time costs: %s' % (t, it, res, (datetime.now() - start_time).total_seconds()))
