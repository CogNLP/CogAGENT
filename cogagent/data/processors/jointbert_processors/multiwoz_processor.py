from platform import processor
import re
import numpy as np
import torch
import random
from torch.nn.modules.container import T
from transformers import BertTokenizer
from transformers import BasicTokenizer
from transformers import WordpieceTokenizer
from collections import Counter

from cogagent.data.datable import DataTable
from cogagent.data.datableset import DataTableSet
from tqdm import tqdm
import transformers
from cogagent.data.processors.base_processor import BaseProcessor
import math
transformers.logging.set_verbosity_error()  # set transformers logging level

import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from collections import Counter


class MultiwozProcessor(BaseProcessor):
    def __init__(self, intent_vocab, tag_vocab, plm):
        """
        :param intent_vocab: list of all intents
        :param tag_vocab: list of all tags
        :param pretrained_weights: which bert, e.g. 'bert-base-uncased'
        """
        self.intent_vocab = intent_vocab
        self.tag_vocab = tag_vocab
        self.intent_dim = len(intent_vocab)
        self.tag_dim = len(tag_vocab)
        self.plm = plm
        self.id2intent = dict([(i, x) for i, x in enumerate(intent_vocab)])
        self.intent2id = dict([(x, i) for i, x in enumerate(intent_vocab)])
        self.id2tag = dict([(i, x) for i, x in enumerate(tag_vocab)])
        self.tag2id = dict([(x, i) for i, x in enumerate(tag_vocab)])
        # dict：无序的，可变的数据集合类型，包括数组不固定的键值对，键具有唯一性
        self.tokenizer = BertTokenizer.from_pretrained(plm)
        self.data = {}
        self.intent_weight = [1] * len(self.intent2id)

    def process(self, data, data_key, cut_sen_len, use_bert_tokenizer=True):
        """
        sample representation: [list of words, list of tags, list of intents, original dialog act]
        :param data_key: train/val/test
        :param data:
        :return:
        """
        datable = DataTable()
        print("Processing data...")
        # print(data[0][0][0])
        # print(np.data.shape())
        self.data[data_key] = data
        max_sen_len, max_context_len = 0, 0
        sen_len = []
        context_len = []
        # print(data[data_key][0][0])
        for d in self.data[data_key][0][0]:
            # print(data[data_key][0])       
            max_sen_len = max(max_sen_len, len(d[0]))
            # print(max_sen_len)
            sen_len.append(len(d[0]))
            # d = (tokens, tags, intents, da2triples(turn["dialog_act"]), context(list of str))
            if cut_sen_len > 0:
                d[0] = d[0][:cut_sen_len]
                d[1] = d[1][:cut_sen_len]
                # d[4] = [' '.join(s.split()[:cut_sen_len]) for s in d[4]]
                d[4] = [' '.join(str(s).split()[:cut_sen_len]) for s in d[4]]

            d[4] = self.tokenizer.encode('[CLS] ' + ' [SEP] '.join(d[4]))
            max_context_len = max(max_context_len, len(d[4]))
            context_len.append(len(d[4]))
            if use_bert_tokenizer:
                word_seq, tag_seq, new2ori = self.bert_tokenize(d[0], d[1])
            else:
                word_seq = d[0]
                tag_seq = d[1]
                # print(tag_seq)
                new2ori = None
            d.append(new2ori)
            d.append(word_seq)
            d.append(self.seq_tag2id(tag_seq))
            d.append(self.seq_intent2id(d[2]))
            # d = (tokens, tags, intents, da2triples(turn["dialog_act"]), context(token id), new2ori, new_word_seq, tag2id_seq, intent2id_seq)
            if data_key=='train':
                for intent_id in d[-1]:
                    self.intent_weight[intent_id] += 1
        if data_key == 'train':
            train_size = len(self.data)
            for intent, intent_id in self.intent2id.items():
                neg_pos = (train_size - self.intent_weight[intent_id]) / self.intent_weight[intent_id]
                self.intent_weight[intent_id] = np.log10(neg_pos)
            self.intent_weight = torch.tensor(self.intent_weight)
        
        print('max sen bert len', max_sen_len)
        print(sorted(Counter(sen_len).items()))
        print('max context bert len', max_context_len)
        print(sorted(Counter(context_len).items()))
        # datable.add_not2torch("new2ori")
        
        # batch_data = random.choices(self.data['train'][0][0], k=100)
        batch_data = self.data[data_key][0][0]
        batch_size = len(batch_data)
        max_seq_len = max([len(x[-3]) for x in batch_data]) + 2
        word_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        tag_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        tag_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        intent_tensor = torch.zeros((batch_size, self.intent_dim), dtype=torch.float)
        context_max_seq_len = max([len(x[-5]) for x in batch_data])
        context_mask_tensor = torch.zeros((batch_size, context_max_seq_len), dtype=torch.long)
        context_seq_tensor = torch.zeros((batch_size, context_max_seq_len), dtype=torch.long)
        for i in range(batch_size):
            words = batch_data[i][-3]
            tags = batch_data[i][-2]
            intents = batch_data[i][-1]
            words = ['[CLS]'] + words + ['[SEP]']
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(str(words))
            sen_len = len(words)
            word_seq_tensor[i][:sen_len] = torch.LongTensor([indexed_tokens])
            tag_seq_tensor[i][1:sen_len-1] = torch.LongTensor(tags)
            word_mask_tensor[i][:sen_len] = torch.LongTensor([1] * sen_len)
            tag_mask_tensor[i][1:sen_len-1] = torch.LongTensor([1] * (sen_len-2))
            for j in intents:
                intent_tensor[i][j] = 1.
            context_len = len(batch_data[i][-5])
            context_seq_tensor[i][:context_len] = torch.LongTensor([batch_data[i][-5]])
            context_mask_tensor[i][:context_len] = torch.LongTensor([1] * context_len)
            
            datable("word_seq_tensor", word_seq_tensor[i])
            datable("word_mask_tensor", word_mask_tensor[i])
            datable("tag_seq_tensor", tag_seq_tensor[i])
            datable("tag_mask_tensor", tag_mask_tensor[i])
            datable("intent_tensor", intent_tensor[i])
            datable("context_seq_tensor", context_seq_tensor[i])
            datable("context_mask_tensor", context_mask_tensor[i])
        # datable.add_not2torch("word_seq_tensor")
        # datable.add_not2torch("word_mask_tensor")
        # datable.add_not2torch("intent2id")
        # datable.add_not2torch("context_mask_tensor")
        return DataTableSet(datable)
        

    def bert_tokenize(self, word_seq, tag_seq):
        split_tokens = []
        new_tag_seq = []
        new2ori = {}
        
        basic_tokens = self.tokenizer.basic_tokenizer.tokenize(' '.join(word_seq))
        # basic_tokens = self.tokenizer.basic_tokenizer.tokenize(' '.join(str(word_seq)))
        accum = ''
        i, j = 0, 0
        for i, token in enumerate(basic_tokens):
            if (str(accum) + str(token)).lower() == str(word_seq[j]).lower():
                accum = ''
            else:
                accum += str(token)
            for sub_token in self.tokenizer.wordpiece_tokenizer.tokenize(str(basic_tokens[i])):
                new2ori[len(new_tag_seq)] = j
                split_tokens.append(sub_token)
                new_tag_seq.append(tag_seq[j])
            if accum == '':
                j += 1
        return split_tokens, new_tag_seq, new2ori

    def seq_tag2id(self, tags):
        return [self.tag2id[x] if x in self.tag2id else self.tag2id['O'] for x in tags]


    def seq_id2tag(self, ids):
        return [self.id2tag[x] for x in ids]

    def seq_intent2id(self, intents):
        return [self.intent2id[x] for x in intents if x in self.intent2id]

    def seq_id2intent(self, ids):
        return [self.id2intent[x] for x in ids]

    def recover_intent(self, intent_logits, slot_logits,tag_mask_tensor,ori_word_seq, new2ori):
        # ori_word_seq = self.data['val'][0]
        # pred_intents = []
        
        max_seq_len = slot_logits.size(0)
        intents = []
        for j in range(self.intent_dim):
            if intent_logits[j] > 0:
                intent, slot, value = re.split('[+*]', self.id2intent[j])
                # intent = self.intent2id[j]
                intents.append([intent, slot, value])
                # intents.append([intent])

        tags = []
        for j in range(1, max_seq_len-1):
            if tag_mask_tensor[j] == 1:
                value, tag_id = torch.max(slot_logits[j], dim=-1)
                tags.append(self.id2tag[tag_id.item()])
        recover_tags = []
        # for t, tag in enumerate(tags):
        #     if int(new2ori[t]) >= len(recover_tags):
        #         recover_tags.append(tag)
        #     # # tag_intent = tag2triples(ori_word_seq, recover_tags)
        ori_word_seq = ori_word_seq[:len(recover_tags)]
        assert len(ori_word_seq)==len(recover_tags)
        tag_intent = []
        t = 0
        while t < len(recover_tags):
            tag = recover_tags[t]
            if tag.startswith('B'):
                intent, slot = tag[2:].split('+')
                value = ori_word_seq[t]
                j = t + 1
                while j < len(recover_tags):
                    if recover_tags[j].startswith('I') and recover_tags[j][2:] == tag[2:]:
                        value += ' ' + ori_word_seq[j]
                        t += 1
                        j += 1
                    else:
                        break
                tag_intent.append([intent, slot, value])
                # tag_intent.append([intent])
            t += 1
        intents += tag_intent

        # intents.append(intents)
        return intents

    
if __name__ == "__main__":
    
    print("处理数据集完成")
