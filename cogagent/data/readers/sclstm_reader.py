import os
from cogagent.data.readers.base_reader import BaseReader
from cogagent.data.datable import DataTable
from cogagent.utils.vocab_utils import Vocabulary

import json
import random
import sys

import torch
from torch.autograd import Variable


class sclstm_multiwoz_reader(BaseReader):
    def __init__(self, raw_data_path, use_cuda=False):
        super(sclstm_multiwoz_reader, self).__init__()
        # setup
        self.feat_file = 'feat.json'
        self.text_file = 'text.json'
        self.vocab_file = 'vocab.txt'
        self.template_file = 'template.txt'
        self.datasplit_file = 'Boo_ResDataSplitRand0925.json'

        self.feat_file_path = os.path.join(raw_data_path, self.feat_file)
        self.text_file_path = os.path.join(raw_data_path, self.text_file)
        self.vocab_file_path = os.path.join(raw_data_path, self.vocab_file)
        self.template_file_path = os.path.join(raw_data_path, self.template_file)
        self.datasplit_file_path = os.path.join(raw_data_path, self.datasplit_file)

        self.USE_CUDA = use_cuda

        # hyper-params
        self.batch_size = 64
        self.max_length = 64
        #        self.percentage = percentage  # percentage of data used
        self.data = {'train': [], 'valid': [], 'test': []}
        self.data_index = {'train': 0, 'valid': 0, 'test': 0}  # index for accessing data
        self.train_data_length = {}
        self.test_data_length = {}
        self.valid_data_length = {}
        self.n_batch = {}
        self.shuffle = True

        # load vocab from file
        self.read_vocab(self.vocab_file_path)  # a list of vocab, andy

        # set input feature cardinality
        self._setCardinality(self.template_file_path)
        self.do_size = self.dfs[1] - self.dfs[0]
        self.da_size = self.dfs[2] - self.dfs[1]
        self.sv_size = self.dfs[3] - self.dfs[2]

        # initialise dataset
        # self._setupData(self.text_file, self.feat_file, self.dataSplit_file)
        # self.reset()

    #  读取数据
    def _read(self, path=None):
        print("Reading data...")

        return self._read(path)

    #  训练集数据读取
    def _read_train(self, text_file_path=None, feat_file_path=None, dataspilt_file_path=None):
        print("Reading train data...")
        datable = DataTable()
        with open(text_file_path) as file:
            dial2text = json.load(file)
        with open(feat_file_path) as file:
            dial2meta = json.load(file)
        with open(dataspilt_file_path) as file:
            dataSet_split = json.load(file)

        data_type = 'train'
        for dial_idx, turn_idx, _ in dataSet_split[data_type]:
            # might have empty feat turn which is not in feat file
            if turn_idx not in dial2meta[dial_idx]:
                continue

            meta = dial2meta[dial_idx][turn_idx]
            text = dial2text[dial_idx][turn_idx]
            datable("meta", meta)
            datable("text", text)
            self.data[data_type].append((dial_idx, turn_idx, text, meta))
            # datable("data", self.data[data_type])
            datable("data_train", self.data[data_type])

        # 记录训练数据长度
        self.train_data_length = len(self.data['train'])
        # setup number of batch
        self.n_batch['train'] = len(self.data['train']) // self.batch_size

        return datable

    #  验证集数据读取
    def _read_dev(self, text_file_path=None, feat_file_path=None, dataspilt_file_path=None):
        print("Reading valid data...")
        datable = DataTable()
        with open(text_file_path) as file:
            dial2text = json.load(file)
        with open(feat_file_path) as f:
            dial2meta = json.load(f)
        with open(dataspilt_file_path) as f:
            dataSet_split = json.load(f)

        data_type = 'valid'
        for dial_idx, turn_idx, _ in dataSet_split[data_type]:
            # might have empty feat turn which is not in feat file
            if turn_idx not in dial2meta[dial_idx]:
                continue

            meta = dial2meta[dial_idx][turn_idx]
            text = dial2text[dial_idx][turn_idx]
            datable("meta", meta)
            datable("text", text)
            self.data[data_type].append((dial_idx, turn_idx, text, meta))
            # datable("data", self.data[data_type])
            datable("data_valid", self.data[data_type])

        # 记录数据验证数据长度
        self.valid_data_length = len(self.data['valid'])
        # setup number of batch
        self.n_batch['valid'] = len(self.data['valid']) // self.batch_size

        return datable

    #  测试集数据读取
    def _read_test(self, text_file_path=None, feat_file_path=None, dataspilt_file_path=None):
        print("Reading text data...")
        datable = DataTable()
        with open(text_file_path) as file:
            dial2text = json.load(file)
        with open(feat_file_path) as file:
            dial2meta = json.load(file)
        with open(dataspilt_file_path) as file:
            dataSet_split = json.load(file)

        data_type = 'test'
        for dial_idx, turn_idx, _ in dataSet_split[data_type]:
            # might have empty feat turn which is not in feat file
            if turn_idx not in dial2meta[dial_idx]:
                continue

            meta = dial2meta[dial_idx][turn_idx]
            text = dial2text[dial_idx][turn_idx]
            datable("meta", meta)
            datable("text", text)
            self.data[data_type].append((dial_idx, turn_idx, text, meta))
            # datable("data", self.data[data_type])
            datable("data_test", self.data[data_type])

        # 记录测试数据长度
        self.test_data_length = len(self.data['test'])

        # setup number of batch
        self.n_batch['test'] = len(self.data['test']) // self.batch_size

        return datable

    #  读取所有数据集
    def read_all(self):
        return self._read_train(self.text_file_path, self.feat_file_path, self.datasplit_file_path), self._read_dev(
            self.text_file_path, self.feat_file_path, self.datasplit_file_path), self._read_test(self.text_file_path,
                                                                                                 self.feat_file_path,
                                                                                                 self.datasplit_file_path)

    #  获取词表
    def read_vocab(self, vocab_file_path):
        # load vocab
        self.word2index = {}
        self.index2word = {}
        idx = 0
        with open(vocab_file_path) as fin:
            for word in fin.readlines():
                word = word.strip().split('\t')[0]
                self.word2index[word] = idx
                self.index2word[idx] = word
                idx += 1
        return {"word2index": self.word2index, "index2word": self.index2word}

    #  重排序数据集
    def reset(self):
        self.data_index = {'train': 0, 'valid': 0, 'test': 0}
        if self.shuffle:
            random.shuffle(self.data['train'])

    def next_batch(self, data_type='train'):
        def indexes_from_sentence(sentence, add_eos=False):
            indexes = [self.word2index[word] if word in self.word2index else self.word2index['UNK_token'] for word in
                       sentence.split(' ')]
            if add_eos:
                return indexes + [self.word2index['EOS_token']]
            else:
                return indexes

        # Pad a with the PAD symbol
        def pad_seq(seq, max_length):
            if max_length > len(seq):
                seq += [self.word2index['PAD_token'] for i in range(max_length - len(seq))]
            else:
                seq = seq[:max_length]
            return seq

        # turn list of word indexes into 1-hot matrix
        def getOneHot(indexes):
            res = []
            for index in indexes:
                hot = [0] * len(self.word2index)
                hot[index] = 1
                res.append(hot)
            return res

        # reading a batch
        start = self.data_index[data_type]
        end = self.data_index[data_type] + 1                   #  取1个数据
        # end = self.data_index[data_type] + self.batch_size   #  取一个batch size的数据
        data = self.data[data_type][start:end]
        self.data_index[data_type] += 1                        # 取1个数据
        #self.data_index[data_type] += self.batch_size         # 取1个batch size的数据

        #sentences, refs, feats, featStrs = [], [], [], []
        #		do_label, da_label, sv_label, sv_seqs = [], [], [], []
        sentences = []
        refs_tuple = []
        feats = []
        featStrs_tuple = []
        sv_indexes_tupel = []
        sv_indexes = []

        for dial_idx, turn_idx, text, meta in data:
            text_ori, text_delex = text['ori'], text['delex']
            sentences.append(indexes_from_sentence(text_delex, add_eos=True))
            refs_tuple.append(text_delex)

            # get semantic feature
            do_idx, da_idx, sv_idx, featStr = self.getFeatIdx(meta)
            do_cond = [1 if i in do_idx else 0 for i in range(self.do_size)]  # domain condition
            da_cond = [1 if i in da_idx else 0 for i in range(self.da_size)]  # dial act condition
            sv_cond = [1 if i in sv_idx else 0 for i in range(self.sv_size)]  # slot/value condition
            feats.append(do_cond + da_cond + sv_cond)
            featStrs_tuple.append(featStr)

            #			# get labels for da, slots
            sv_indexes_tupel.append(sv_idx)

        # Zip into pairs, sort by length (descending), unzip
        # Note: _words and _seqs should be sorted in the same order
        seq_pairs = sorted(zip(sentences, refs_tuple, feats, featStrs_tuple, sv_indexes_tupel), key=lambda p: len(p[0]), reverse=True)
        sentences, refs_tuple, feats, featStrs_tuple, sv_indexes_tupel = zip(*seq_pairs)
        sv_indexes = sv_indexes_tupel[0]
        refs = refs_tuple[0]
        featStrs = featStrs_tuple[0]

        # Pad with 0s to max length
        lengths_list = [len(s) for s in sentences]
        sentences_padded = [pad_seq(s, self.max_length) for s in sentences]

        # Turn (batch_size, max_len) into (batch_size, max_len, n_vocab)
        sentences = [getOneHot(s) for s in sentences_padded]

        input_var = Variable(torch.FloatTensor(sentences))
        label_var = Variable(torch.LongTensor(sentences_padded))
        feats_var = Variable(torch.FloatTensor(feats))

        if self.USE_CUDA:
            input_var = input_var.cuda()
            label_var = label_var.cuda()
            feats_var = feats_var.cuda()

        lengths = lengths_list[0]

        return input_var, label_var, feats_var, lengths, refs, featStrs, sv_indexes

    #  input_var : 1 * n * 1392

    def _setCardinality(self, template_file_path):
        self.cardinality = []
        with open(template_file_path) as f:
            self.dfs = [0, 0, 0, 0]
            for line in f.readlines():
                self.cardinality.append(line.replace('\n', ''))
                if line.startswith('d:'):
                    self.dfs[1] += 1
                elif line.startswith('d-a:'):
                    self.dfs[2] += 1
                elif line.startswith('d-a-s-v:'):
                    self.dfs[3] += 1
            for i in range(0, len(self.dfs) - 1):
                self.dfs[i + 1] = self.dfs[i] + self.dfs[i + 1]

    def printDataInfo(self):
        print('***** DATA INFO *****')
        print('Using {}% of training data'.format(self.percentage * 100))
        print('BATCH SIZE:', self.batch_size)

        print('Train:', len(self.data['train']), 'turns')
        print('Valid:', len(self.data['valid']), 'turns')
        print('Test:', len(self.data['test']), 'turns')
        print('# of turns', file=sys.stderr)
        print('Train:', len(self.data['train']), file=sys.stderr)
        print('Valid:', len(self.data['valid']), file=sys.stderr)
        print('Test:', len(self.data['test']), file=sys.stderr)
        print('# of batches: Train {} Valid {} Test {}'.format(self.n_batch['train'], self.n_batch['valid'],
                                                               self.n_batch['test']))
        print('# of batches: Train {} Valid {} Test {}'.format(self.n_batch['train'], self.n_batch['valid'],
                                                               self.n_batch['test']), file=sys.stderr)
        print('*************************\n')

    def _setupData(self, text_file, feat_file, dataSplit_file):
        with open(text_file) as f:
            dial2text = json.load(f)
        with open(feat_file) as f:
            dial2meta = json.load(f)
        with open(dataSplit_file) as f:
            dataSet_split = json.load(f)

        for data_type in ['train', 'valid', 'test']:
            for dial_idx, turn_idx, _ in dataSet_split[data_type]:
                # might have empty feat turn which is not in feat file
                if turn_idx not in dial2meta[dial_idx]:
                    continue

                meta = dial2meta[dial_idx][turn_idx]
                text = dial2text[dial_idx][turn_idx]
                self.data[data_type].append((dial_idx, turn_idx, text, meta))

        # percentage of training data
        if self.percentage < 1:
            _len = len(self.data['train'])
            self.data['train'] = self.data['train'][:int(_len * self.percentage)]

        # setup number of batch
        for _type in ['train', 'valid', 'test']:
            self.n_batch[_type] = len(self.data[_type]) // self.batch_size
            self.n_batch['train']

        self.printDataInfo()

    def getFeatIdx(self, meta):
        feat_container = []
        do_idx, da_idx, sv_idx = [], [], []
        for da, slots in meta.items():
            do = da.split('-')[0]
            _do_idx = self.cardinality.index('d:' + do) - self.dfs[0]
            if _do_idx not in do_idx:
                do_idx.append(_do_idx)
            da_idx.append(self.cardinality.index('d-a:' + da) - self.dfs[1])
            for _slot in slots:  # e.g. ('Day', '1', 'Wednesday ')
                sv_idx.append(self.cardinality.index('d-a-s-v:' + da + '-' + _slot[0] + '-' + _slot[1]) - self.dfs[2])
                feat_container.append(da + '-' + _slot[0] + '-' + _slot[1])

        feat_container = sorted(feat_container)  # sort SVs across DAs to make sure universal order
        feat = '|'.join(feat_container)

        return do_idx, da_idx, sv_idx, feat


class SimpleDatasetWoz(sclstm_multiwoz_reader):
    def __init__(self, config):
        raw_data_path = "/home/nlp/CogAGENT/datapath/sclstm_multiwoz_data/resource"
        vocab_file = config['DATA']['vocab_file']
        template_file = config['DATA']['template_file']
        self.vocab_file_path = os.path.join(raw_data_path, vocab_file)
        self.batch_size = 1

        # load vocab from file
        self.read_vocab(self.vocab_file_path)  # a list of vocab, andy

        # set input feature cardinality
        self._setCardinality(template_file)
        self.do_size = self.dfs[1] - self.dfs[0]
        self.da_size = self.dfs[2] - self.dfs[1]
        self.sv_size = self.dfs[3] - self.dfs[2]


if __name__ == "__main__":
    reader = sclstm_multiwoz_reader(raw_data_path="/home/nlp/CogAGENT/datapath/sclstm_multiwoz_data/resource")
    n_batch_train, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab(vocab_file_path="/home/nlp/CogAGENT/datapath/sclstm_multiwoz_data/resource/vocab.txt")
    print("end")
