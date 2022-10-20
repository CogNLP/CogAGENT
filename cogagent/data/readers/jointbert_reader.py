import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import zipfile
import json
from cogagent.data.readers.base_reader import BaseReader
from cogagent.data.datable import DataTable
from cogagent.utils.vocab_utils import Vocabulary
from collections import Counter
# from cogagent.utils.download_utils import Downloader



class JointbertReader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        self.train_file = 'train_data.json'
        # self.train_file = 'train_data2.json'
        self.dev_file = 'val_data.json'
        # self.dev_file = 'val_data2.json'
        self.test_file = 'test_data.json'
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)

    def _read(self, path=None):
        
        data = {}
        if path == self.train_path:
            print("Reading train data...")
            with open(self.train_path, encoding='utf-8') as f:# 读取文件
                data = json.load(f)
            print('{}, size {}'.format(self.train_file, len(data)))
            datable = DataTable()
            datable('train', data)
            # return data['train']
        elif path == self.dev_path:
            print("Reading dev data...")
            with open(self.dev_path, encoding='utf-8') as f:
                data = json.load(f)
            print('{}, size {}'.format(self.dev_file, len(data)))
            datable = DataTable()
            datable('val', data)
            # return data['val']
        else:
            print("Reading test data...")
            with open(self.test_path, encoding='utf-8') as f:
                data = json.load(f)
            print('{}, size {}'.format(self.test_file, len(data)))
            datable = DataTable()
            datable('test', data)
            # return data['test']

        
        return datable
        # return data2

    def _read_train(self, path=None):
        return self._read(path)

    def _read_dev(self, path=None):
        return self._read(path)

    def _read_test(self, path=None):
        return self._read(path)

    def read_all(self):
        return self._read_train(self.train_path), self._read_dev(self.dev_path), self._read_test(self.test_path)

    def read_vocab(self):
        pass

    def read_addition(self):
        pass


if __name__ == "__main__":
    reader = JointbertReader(raw_data_path="/home/nlp/anaconda3/envs/CogAGENT/CogAGENT/datapath/nlu/multiwoz/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    print("读取数据完成")