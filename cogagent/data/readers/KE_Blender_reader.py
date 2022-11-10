import os
from cogagent.data.readers.base_reader import BaseReader
from cogagent.data.datable import DataTable
from cogagent.utils.vocab_utils import Vocabulary
import pandas as pd
import logging
#from seq2seq_model import Seq2SeqModel
import json
import re
import torch
import numpy as np
import random


# from cogagent.utils.download_utils import Downloader


class KE_BlenderReader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        # downloader = Downloader()
        # downloader.download_sst2_raw_data(raw_data_path)
        self.train_file = 'train.json'
        self.dev_file = 'valid_topic_split.json'
        self.test_file = 'test_topic_split.json'
        self.train_sim_file = 'train_sim.txt'
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)
        self.train_sim_path = os.path.join(raw_data_path, self.train_sim_file)
        self.label_vocab = Vocabulary()

    def _read(self, path=None):
        print("Reading data...")
        datable = DataTable()
        with open(path) as file:
            lines = file.readlines()
        header = lines[0]
        contents = lines[1:]
        for line in contents:
            sentence, label = line.strip().split("\t")
            datable("sentence", sentence)
            datable("label", label)
            self.label_vocab.add(label)
        return datable
    def read_wizard_json(self, file_path):
        with open(file_path, 'r') as f:
            file = json.load(f)

        data = []
        for line in file:
            tmp_source = ''
            for i in line['dialog']:
                utt = i['text']
                if tmp_source != '':
                    data.append([tmp_source, "__start__ " + utt + " __end__"])
                    # add split '\t' for blender
                    tmp_source = tmp_source + "\t" + utt
                else:
                    tmp_source = utt
        return data

    def read_wizard_definition(self, file_path):

        with open(file_path, 'r') as f:
            file = json.load(f)

        data = []
        for line in file:
            # line.keys() ['chosen_topic', 'persona', 'wizard_eval', 'dialog', 'chosen_topic_passage']
            # dialog.keys() dict_keys(['speaker', 'text', 'candidate_responses', 'retrieved_passages', 'retrieved_topics'])
            for i in line['dialog']:
                utt = i['text']
                external_passage = i['retrieved_passages']
                for j in external_passage:
                    if list(j.keys())[0].lower() in utt.lower():
                        know_key = list(j.keys())[0]
                        try:
                            # retrieved knowledge is available
                            # we do not use gold knowledge
                            utt_mask = re.sub(know_key, '[MASK]', utt, flags=re.IGNORECASE)
                            knowledge = ('\t').join(j[know_key])
                            data.append([utt_mask, "__defi__ " + knowledge + " __end__"])
                        except:
                            continue
        return data

    def read_wizard_concat_json(self, file_path):

        with open(file_path, 'r') as f:
            file = json.load(f)

            # print(file[0].keys())
            # print(file[0]['dialog'])
        data = []
        for line in file:

            tmp_source = ''
            for i in line['dialog']:
                utt = i['text']

                external_passage = i['retrieved_passages']
                for j in external_passage:
                    # print(list(j.keys()))
                    if list(j.keys())[0].lower() in utt.lower():
                        # retrieved knowledge is available
                        know_key = list(j.keys())[0]
                        knowledge = (" ").join(j[know_key])
                        if tmp_source != '':
                            data.append([tmp_source + " " + knowledge, "__start__ " + utt + " __end__"])
                            tmp_source = tmp_source + "\t" + utt
                        else:
                            tmp_source = utt
                        break
        return data

    def read_hypernym(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            file = f.readlines()
        for line in file:
            source, target = line.strip().split('\001')
            data.append([source, "__hype__ " + target + " __end__"])
        return data

    def _read_train(self, path=None):
        print("Reading train data...")
        datable = DataTable()
        # data = []
        train_data = self.read_wizard_json(self.train_path) + self.read_wizard_definition(self.train_path) + \
                     self.read_hypernym(self.train_sim_path) + self.read_wizard_concat_json(self.train_path)
        # for i in range(len(train_data)):
        for i in range(500):
            # data.append(train_data[i])
            datable("train_data", train_data[i])
        return datable

    def _read_dev(self, path=None):
        print("Reading valid data...")
        datable = DataTable()
        # data = []
        dev_data = self.read_wizard_json(self.dev_path)
        # for i in range(len(dev_data)):
        for i in range(500):
            # data.append(dev_data[i])
            datable("dev_data", dev_data[i])
        return datable

    def _read_test(self, path=None):
        print("Reading test data...")
        datable = DataTable()
        # data = []
        test_data = self.read_wizard_json(self.test_path)
        # for i in range(len(test_data)):
        for i in range(500):
            # data.append(test_data[i])
            datable("test_data", test_data[i])
        return datable

    def read_all(self):
        return self._read_train(self.train_path), self._read_dev(self.dev_path), self._read_test(self.test_path)

    def read_vocab(self):
        self.label_vocab.create()
        return {"label_vocab": self.label_vocab}






if __name__ == "__main__":
    reader = KE_BlenderReader(raw_data_path="/home/nlp/CogAGENT/datapath/KE_Blender_data/")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    print("end")
