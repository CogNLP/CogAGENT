import sys

import torch
import torch.nn as nn
from torch.autograd import Variable

from cogagent.models.decoder_deep import DecoderDeep
from cogagent.core.loss.masked_corss_entropy import masked_cross_entropy
from cogagent.data.readers.sclstm_reader import sclstm_multiwoz_reader
from cogagent.models.base_model import BaseModel

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

class LMDeep(BaseModel):
	def __init__(self, dec_type, input_size, output_size, hidden_size, d_size, n_layer=1, dropout=0.5, lr=0.001, use_cuda=True):
		super(LMDeep, self).__init__()
		self.dec_type = dec_type
		self.hidden_size = hidden_size
		print('Using deep version with {} layer'.format(n_layer))
		print('Using deep version with {} layer'.format(n_layer), file=sys.stderr)
		self.USE_CUDA = use_cuda
		self.dec = DecoderDeep(dec_type, input_size, output_size, hidden_size, d_size=d_size, n_layer=n_layer, dropout=dropout, use_cuda=use_cuda).to(device)
#		if self.dec_type != 'sclstm':
#			self.feat2hidden = nn.Linear(d_size, hidden_size)
		self.dataset = sclstm_multiwoz_reader(raw_data_path="/home/nlp/CogAGENT/datapath/sclstm_multiwoz_data/resource")

		self.set_solver(lr)

	def	forward(self, input_var, feats_var, gen=False, beam_search=False, beam_size=1):
		batch_size = self.dataset.batch_size
		#batch_size = dataset.batch_size
		if self.dec_type == 'sclstm':
			init_hidden = Variable(torch.zeros(batch_size, self.hidden_size))
			if self.USE_CUDA:
				init_hidden = init_hidden.to(device)
			'''
			train/valid (gen=False, beam_search=False, beam_size=1)
	 		test w/o beam_search (gen=True, beam_search=False, beam_size=beam_size)
	 		test w/i beam_search (gen=True, beam_search=True, beam_size=beam_size)
			'''
			if beam_search:
				assert gen
				# self dataset 替换
				decoded_words = self.dec.beam_search(input_var, self.dataset, init_hidden=init_hidden, init_feat=feats_var, \
														gen=gen, beam_size=beam_size)
				return decoded_words  # list (batch_size=1) of list (beam_size) with generated sentences

			# w/o beam_search
			sample_size = beam_size
			decoded_words = [ [] for _ in range(batch_size) ]
			for sample_idx in range(sample_size): # over generation
				# 结果是self.output_prob, decoded_words  获取概率分布预测，和生成解码词汇
				self.output_prob, gens = self.dec(input_var, self.dataset, init_hidden=init_hidden, init_feat=feats_var, gen=gen, sample_size=sample_size)
				self.output_prob = self.output_prob.to(device)


				for batch_idx in range(batch_size):
					decoded_words[batch_idx].append(gens[batch_idx])  #  将每个解码词加入

			return decoded_words, self.output_prob  # list (batch_size) of list (sample_size) with generated sentences


		#  else: # TODO: vanilla lstm
			#  pass
#			last_hidden = self.feat2hidden(conds_batches)
#			self.output_prob, decoded_words = self.dec(input_seq, dataset, last_hidden=last_hidden, gen=gen, random_sample=self.random_sample)


	def generate(self, feats_var, beam_size=1):
		batch_size = self.dataset.batch_size
		init_hidden = Variable(torch.zeros(batch_size, self.hidden_size))
		if self.USE_CUDA:
			init_hidden = init_hidden.to(device)
		decoded_words = self.dec.beam_search(None, self.dataset, init_hidden=init_hidden, init_feat=feats_var, \
														gen=True, beam_size=beam_size)
		return decoded_words

	def set_solver(self, lr):
		if self.dec_type == 'sclstm':
			self.solver = torch.optim.Adam(self.dec.parameters(), lr=lr)
		else:
			self.solver = torch.optim.Adam([{'params': self.dec.parameters()}, {'params': self.feat2hidden.parameters()}], lr=lr)

	def loss(self, batch, loss_function):
		input_var = batch["train_input_var"]
		feats_var = batch["train_feats_var"]
		pred, output_prob = self.forward(input_var, feats_var,)  #  output_prob:torch.Size([64, 64, 1392])
		# pred = self.forward(batch)
		loss = loss_function.forward(output_prob, batch["train_label_var"], batch["train_lengths"])
		# loss = loss / (3734/self.dataset.batch_size)
		return loss

	# def get_loss(self, target_label, target_lengths):
	# 	self.loss = masked_cross_entropy(
	# 		self.output_prob.contiguous(),  # -> batch x seq
	# 		target_label.contiguous(),  # -> batch x seq
	# 		target_lengths)
	# 	return self.loss
	def dev_loss(self, batch, loss_function):
		input_var = batch["dev_input_var"]
		feats_var = batch["dev_feats_var"]
		pred, output_prob = self.forward(input_var, feats_var,)  #  output_prob:torch.Size([64, 64, 1392])
		# pred = self.forward(batch)
		loss = loss_function.forward(output_prob, batch["dev_label_var"], batch["dev_lengths"])
		return loss

	def evaluate(self, batch, metric_function):

		countBatch, countPerGen, loss = self.predict(batch)
		metric_function.evaluate(countBatch, countPerGen, loss)

		# tatal_loss = 0
		# decoded_words = self.forward(batch["dev_input_var"], batch["dev_feats_var"], gen=False, beam_size=False, beam_search=1)
		# loss = self.get_loss(batch["dev_label_var"], batch["dev_lengths"])
		# total_loss += loss.item()
		# decoded_words = self.forward(batch["dev_input_var"], batch["dev_feats_var"], gen=True, beam_size=False, beam_search=1)
		# CountBatch, countPerGen = get_slot_error(dataset, decoded_words, refs, sv_indexes)

	def predict(self, batch):

		decoded_words, output_prob = self.forward(batch["dev_input_var"], batch["dev_feats_var"],
														gen=False, beam_search=False, beam_size=1)

		# update loss
		loss = self.get_loss(output_prob, batch["dev_label_var"], batch["dev_lengths"])
		# loss = model.get_loss(label_var, lengths)

		loss = loss.item()
        # total_loss += loss.item()

		# run generation for calculating slot error
		decoded_words, output_prob = self.forward(batch["dev_input_var"], batch["dev_feats_var"],
														gen=True, beam_search=False, beam_size=1)
		# decoded_words = model(input_var, dataset, feats_var, gen=True, beam_search=False, beam_size=1)
		countBatch, countPerGen = self.get_slot_error(decoded_words, batch["dev_refs"], batch["dev_sv_indexes"])
		return countBatch, countPerGen, loss


	def get_loss(self, output_prob, target_label, target_lengths):
		loss = masked_cross_entropy(
			output_prob.contiguous(), # -> batch x seq
			target_label.contiguous(), # -> batch x seq
			target_lengths)
		return loss

	def score(self, feat, gen, template):
		'''
		feat = ['d-a-s-v:Booking-Book-Day-1', 'd-a-s-v:Booking-Book-Name-1', 'd-a-s-v:Booking-Book-Name-2']
		gen = 'xxx slot-booking-book-name xxx slot-booking-book-time'
		'''
		das = []  # e.g. a list of d-a-s-v:Booking-Book-Day
		with open(template) as f:
			for line in f:
				if 'd-a-s-v:' not in line:
					continue
				if '-none' in line or '-?' in line or '-yes' in line or '-no' in line:
					continue
				tok = '-'.join(line.strip().split('-')[:-1])
				if tok not in das:
					das.append(tok)

		total, redunt, miss = 0, 0, 0
		for _das in das:
			feat_count = 0
			das_order = [_das + '-' + str(i) for i in range(20)]
			for _feat in feat:
				if _feat in das_order:
					feat_count += 1
			_das = _das.replace('d-a-s-v:', '').lower().split('-')
			slot_tok = '@' + _das[0][:3] + '-' + _das[1] + '-' + _das[2]

			gen_count = gen.split().count(slot_tok)
			diff_count = gen_count - feat_count
			if diff_count > 0:
				redunt += diff_count
			else:
				miss += -diff_count
			total += feat_count
		return total, redunt, miss

	def get_slot_error(self, gens, refs, sv_indexes):
		'''
		Args:
			gens:  (batch_size, beam_size)
			refs:  (batch_size,)
			sv:    (batch_size,)
		Returns:
			count: accumulative slot error of a batch
			countPerGen: slot error for each sample
		'''
		batch_size = len(gens)  # batch_size 64
		beam_size = len(gens[0])  #  beam_size  1
		assert len(refs) == batch_size and len(sv_indexes) == batch_size

		count = {'total': 0.0, 'redunt': 0.0, 'miss': 0.0}
		countPerGen = [[] for _ in range(batch_size)]
		for batch_idx in range(batch_size):
			for beam_idx in range(beam_size):
				# c = self.dataset.dfs[2]  #  self.dataset.dfs[2]是int
				# a = [x for x in sv_indexes[batch_idx]]
				#
				# for x in sv_indexes[batch_idx]:
				# 	if x[0]:
				# 		a = x[0] + self.dataset.dfs[2]
				# 		b = self.dataset.cardinality[a]
				# 		felements = [b]
				# 	else:
				# 		a = x + self.dataset.dfs[2]
				# 		b = self.dataset.cardinality[a]
				# 		felements = [b]

				felements = [self.dataset.cardinality[x+self.dataset.dfs[2]] for x in sv_indexes[batch_idx]]

				# get slot error per sample(beam)
				total, redunt, miss = self.score(felements, gens[batch_idx][beam_idx], self.dataset.template_file_path)

				c = {}
				for a, b in zip(['total', 'redunt', 'miss'], [total, redunt, miss]):
					c[a] = b
					count[a] += b
				countPerGen[batch_idx].append(c)

		return count, countPerGen

	def update(self, clip):
		# Back prop
		self.loss.backward()

		# Clip gradient norms
		_ = torch.nn.utils.clip_grad_norm(self.dec.parameters(), clip)

		# Update
		self.solver.step()

		# Zero grad
		self.solver.zero_grad()

class get_loss(nn.Module):
	def __init__(self, ):
		super(get_loss, self).__init__()

	def forward(self, output_prob, target_label, target_lengths):
		self.loss = masked_cross_entropy(output_prob, target_label, target_lengths)
		return self.loss



