import argparse
import random
from cogagent.data.datable import DataTable
from cogagent.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import transformers
from cogagent.data.processors.base_processor import BaseProcessor
import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
transformers.logging.set_verbosity_error()  # set transformers logging level
EOS_token = 1
PAD_token = 3
class PolicyProcessor(BaseProcessor):
    def __init__(self, input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index):
        super().__init__()
        self.output_lang_index2word = output_lang_index2word
        self.input_lang_index2word = input_lang_index2word

        self.output_lang_word2index = output_lang_word2index
        self.input_lang_word2index = input_lang_word2index

    def _process(self, data, data_key):
        datable = DataTable()
        print("Processing",data_key)
        data2 = data.datas[data_key][0]
        
        # dials = list(data2.keys())
        # random.shuffle(dials)
        
        i = 0
        for name, val_file in data2.items():
            # input_id = [];target_id = [];bs_tensor = [];db_tensor = [];
            input_tensor = [];target_tensor = [];bs_id = [];db_id = []
            if i == 30:break
            # val_file = data2[name]
            # 每个json即每条数据
            for idx, (usr, sys, bs, db) in enumerate(zip(val_file['usr'], val_file['sys'], val_file['bs'], val_file['db'])):
                tensor = [self.input_word2index(word) for word in usr.strip(' ').split(' ')] + [EOS_token]  # model.input_word2index(word)
                input_tensor.append(torch.LongTensor(tensor))
                # 一个json里面usr包含若干句话的list/tensor表示
                tensor = [self.output_word2index(word) for word in sys.strip(' ').split(' ')] + [EOS_token]
                target_tensor.append(torch.LongTensor(tensor))
                bs_id.append([float(belief) for belief in bs]) 
                db_id.append([float(belief) for belief in db]) 

            input_tensor,input_lengths= self.padSequences(input_tensor, longest_sent=60)
            datable("input_tensor", input_tensor)
            datable("input_lengths", input_lengths)
            target_tensor,target_lengths= self.padSequences(target_tensor, longest_sent=60)
            datable("target_tensor", target_tensor)
            datable("target_lengths", target_lengths)

                # tensor = [float(belief) for belief in bs]
                # tensor_length = len(tensor)
                # bs_tensor = np.ones((94)) * PAD_token
                # bs_tensor[0:tensor_length] = tensor[:tensor_length]
                # bs_tensor = torch.tensor(bs_tensor, dtype=torch.float)
                # datable("bs_tensor", bs_tensor)

                # tensor = [float(pointer) for pointer in db]
                # tensor_length = len(tensor)
                # db_tensor = np.ones((30)) * PAD_token
                # db_tensor[0:tensor_length] = tensor[:tensor_length]
                # db_tensor = torch.tensor(db_tensor, dtype=torch.float)
                # datable("db_tensor", db_tensor)

            # bs_tensor = self.padSequence(bs_tensor, longest_sent=94)
            tensor_lengths = [len(sentence) for sentence in bs_id]
            bs_tensor = np.zeros((32, 94))
            # copy over the actual sequences
            for j, x_len in enumerate(tensor_lengths):
                sequence = bs_id[j]
                bs_tensor[j, 0:x_len] = sequence[:x_len]
            bs_tensor = torch.tensor(bs_tensor, dtype=torch.float)
            datable("bs_tensor", bs_tensor)  
            #  db_tensor  = self.padSequence(db_tensor, longest_sent=30)
            tensor_lengths = [len(sentence) for sentence in db_id]
            db_tensor = np.zeros((32, 30))
            for j, x_len in enumerate(tensor_lengths):
                sequence = db_id[j]
                db_tensor[j, 0:x_len] = sequence[:x_len]
            db_tensor = torch.tensor(db_tensor, dtype=torch.float)
            datable("db_tensor", db_tensor)
    
            i = i + 1
        return DataTableSet(datable)

    def process_train(self, data, data_key):
        return self._process(data, data_key)

    def process_dev(self, data, data_key):
        return self._process(data, data_key)

    def process_test(self, data, data_key):
        return self._process(data, data_key)

    def input_word2index(self, index):
        if index in self.input_lang_word2index:
            return self.input_lang_word2index[index]
        else:
            return 2

    def output_word2index(self, index):
        if index in self.output_lang_word2index:
            return self.output_lang_word2index[index]
        else:
            return 2
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    def padSequences(self, sentence_ids, longest_sent):
        pad_token = PAD_token
        batch_size = 32
        # 对某个json中的某个usr/sys（包含若干句子）进行处理，找到最长的句子，其余补0
        id_lengths = [len(sentence) for sentence in sentence_ids]
        tensor_lengths = np.ones((batch_size), dtype=int)
        tensor_lengths[0:len(id_lengths)] = id_lengths
        # longest_sent = max(tensor_lengths)
        # longest_sent = 60
        # batch_size = len(sentence_ids)
        
        padded_tensor = np.ones((batch_size, longest_sent)) * pad_token

        # copy over the actual sequences
        for i, x_len in enumerate(id_lengths):
            sequence = sentence_ids[i]
            padded_tensor[i, 0:x_len] = sequence[:x_len]

        padded_tensor = torch.LongTensor(padded_tensor)
        return padded_tensor,tensor_lengths

    def padSequence(self, sentence, longest_sent):
        pad_token = PAD_token
        # 对某个json中的某个usr/sys的一个句子进行处理，句子长度定为60，补0
        tensor_length = len(sentence)
        padded_tensor = np.ones((longest_sent)) * pad_token
        # copy over the actual sequences
        padded_tensor[0:tensor_length] = sentence[:tensor_length]
        padded_tensor = torch.LongTensor(padded_tensor)
        return padded_tensor

if __name__ == "__main__":
    print("end")
