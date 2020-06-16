# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-06-06
Description: 训练主程序
"""
import os
import sys
import logging
import logging.config
import time
import yaml
import torch
import torch.optim as optim
import argparse
from constants import ROOT_PATH
from utils.data import Data
from utils.functions import batch_char_sequence_labeling_process, predict_check, evaluate_model
from model.cnn_attn_lstm_crf import CnnAttnLstmCRF


class Run(object):
	def __init__(self, model_config: dict):
		"""

		:param model_config: path
		"""
		self.model_config = model_config
		self.encoder_type = self.model_config['encoder_type']
		alphabet_path = self.model_config['alphabet_path'] + '/' + self.encoder_type + '.dset'
		data_config_file = self.model_config['data_config_file']
		self.data = Data(data_config_file, alphabet_path)
		# 不做替换的lexi_index
		no_replaced_lexicon_name = self.data.data_config['no_replaced_lexicon_name']
		self.no_replaced_lexicon_id = [self.data.lexicon_alphabet.get_index(i) for i in no_replaced_lexicon_name]

	def train(self):
		if self.encoder_type == 'cnn_attn_lstm_crf':
			model = CnnAttnLstmCRF(self.data, self.model_config)
		else:
			print('No model selects')
			return
		logger.info('model config: %s' % self.model_config)
		optimizer = optim.Adam(model.parameters(), lr=self.model_config['lr'], weight_decay=self.model_config['l2'])
		if self.model_config['gpu']:
			model = model.cuda()
		best_dev = -10
		best_dev_loss = 1000000
		last_improved = 0

		for idx in range(self.model_config['epoch']):
			epoch_start = time.time()
			temp_start = epoch_start
			logger.info('Epoch : %s/%s' % (idx, self.model_config['epoch']))
			optimizer = self.lr_decay(optimizer, idx, self.model_config['lr_decay'], self.model_config['lr'])

			sample_loss = 0
			total_loss = 0
			right_token = 0
			whole_token = 0
			logger.info('first input word list: %s' % (self.data.train_texts[0][1]))

			model.train()
			model.zero_grad()
			batch_size = self.model_config['batch_size']
			train_num = len(self.data.train_ids)
			total_batch = train_num // batch_size + 1
			logger.info('total_batch: %s' % total_batch)

			end = 0

			for batch_id in range(total_batch):
				start = batch_id * batch_size
				end = (batch_id + 1) * batch_size
				if end > train_num:
					end = train_num
				instance = self.data.train_ids[start: end]
				if not instance:
					continue
				if self.encoder_type == 'cnn_attn_lstm_crf':
					batch_char, batch_word, batch_intent, batch_lexicon, batch_label, batch_char_len, mask, batch_lexicon_indices, \
						batch_word_indices = batch_char_sequence_labeling_process(
							instance, self.model_config['gpu'], self.data.char_max_length, self.data.word_max_length,
							self.no_replaced_lexicon_id, True)
					loss, tag_seq = model(
						batch_char, batch_word, batch_intent, batch_lexicon, batch_char_len, mask, batch_lexicon_indices,
						batch_word_indices, batch_label)
				else:
					print('No model selects')
					return

				right, whole = predict_check(tag_seq, batch_label, mask)
				right_token += right
				whole_token += whole
				sample_loss += loss.item()
				total_loss += loss.item()

				if end % (batch_size * 10) == 0:
					temp_time = time.time()
					temp_cost = temp_time - temp_start
					temp_start = temp_time
					logger.info("Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
						end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
					if sample_loss > 1e8 or str(sample_loss) == 'nan':
						raise ValueError("ERROR: LOSS EXPLOSION (>1e8)!")
					sample_loss = 0

				loss.backward()
				optimizer.step()
				model.zero_grad()

			temp_time = time.time()
			temp_cost = temp_time - temp_start
			logger.info("Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
				end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
			epoch_finish = time.time()
			epoch_cost = epoch_finish - epoch_start
			logger.info("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
				idx, epoch_cost, train_num / epoch_cost, total_loss))

			# evaluation:
			self.model_config['char_max_length'] = self.data.char_max_length
			self.model_config['word_max_length'] = self.data.word_max_length
			self.model_config['no_replaced_lexicon_id'] = self.no_replaced_lexicon_id
			dev_loss, acc, p, r, f, _, _ = evaluate_model(
				self.data, model, "dev", self.model_config, self.data.label_alphabet, self.encoder_type)
			dev_finish = time.time()
			dev_cost = dev_finish - epoch_finish
			current_score = f
			current_dev_loss = dev_loss
			logger.info("Epoch: %s, Loss: %s, Dev: time: %.2fs, dev_loss: %s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
				idx, total_loss, dev_cost, dev_loss, acc, p, r, f))

			# save model
			if current_score > best_dev:
				model_name = self.model_config['model_path'] + '-' + str(idx) + '.model'
				model_name = os.path.join(ROOT_PATH, model_name)
				torch.save(model.state_dict(), model_name)
				logger.info("Saved Model, Epoch:%s, Current score: %.4f, Last score: %.4f, Last loss: %s" % (
					idx, current_score, best_dev, best_dev_loss))
				best_dev = current_score
				best_dev_loss = current_dev_loss
				last_improved = idx

			test_loss, acc, p, r, f, _, _ = evaluate_model(
				self.data, model, 'test', self.model_config, self.data.label_alphabet, self.encoder_type)
			test_finish = time.time()
			test_cost = test_finish - dev_finish
			logger.info(
				"Epoch: %s, Loss: %s, Test: time: %.2fs, test_loss: %s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
					idx, total_loss, test_cost, test_loss, acc, p, r, f))

			if idx - last_improved > self.model_config['require_improvement']:
				logger.info('No optimization for %s epoch, auto-stopping' % self.model_config['require_improvement'])
				break

	@staticmethod
	def lr_decay(optimizer, epoch, decay_rate, init_lr):
		lr = init_lr / (1 + decay_rate * epoch)
		logging.info("Learning rate is set as: %s", lr)
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
		return optimizer

	@classmethod
	def read_configs(cls, model_config_file):
		with open(model_config_file, 'r') as rf:
			model_config = yaml.load(rf, Loader=yaml.FullLoader)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		gpu = True if device.type == 'cuda' else False
		map_location = 'cpu' if gpu is False else None
		model_config.update({'device': device, 'gpu': gpu, 'map_location': map_location})
		# 读取model_num对应的模型超参数,重复参数会覆盖
		model_num = model_config['model_num']
		for k, v in model_config['model'][model_num].items():
			model_config[k] = v
		del model_config['model']

		return cls(model_config)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--model_config_file',
		type=str,
		default=os.path.join(ROOT_PATH, 'conf/model_config.yaml'),
		help='model config file path.')
	parser.add_argument(
		'--mode',
		type=str,
		default=None,
		choices=['train', 'validation'],
		required=True,
		help='operating mode'
	)
	parser.add_argument(
		'--del_log',
		type=bool,
		default=True,
		help='delete the log'
	)
	arguments = parser.parse_args()

	return arguments


if __name__ == '__main__':
	args = parse_args()

	if args.del_log is True:
		log_file = os.path.join(ROOT_PATH, 'logs/run.log')
		if os.path.exists(log_file):
			os.remove(log_file)
	# train mode
	if args.mode == 'train':
		logger = logging.getLogger(__name__)
		logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
		run = Run.read_configs(args.model_config_file)
		run.train()
	else:
		pass
