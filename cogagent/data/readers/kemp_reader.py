import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import json
from cogagent.data.readers.base_reader import BaseReader
from cogagent.data.datable import DataTable
from cogagent.utils.vocab_utils import Vocabulary
from collections import Counter
# from cogagent.utils.download_utils import Downloader

class KempReader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        self.file = 'kemp_dataset_preproc.json'
        self.datapath = os.path.join(raw_data_path, self.file)
    
    def _read(self, data_key, path=None):
        print("Reading",data_key,"data...")
        with open(self.datapath, 'r') as f:
            data_train, data_val, data_test, vocab = json.load(f)
        
        if data_key == 'train':
            print("train length: ", len(data_train['situation']))
            
        elif data_key == 'val':
            print("valid length: ", len(data_val['situation']))
            
        else:
            print("test length: ", len(data_test['situation']))
        datable = DataTable()
        datable('context', data_train['context'])
        datable('target', data_train['target'])
        datable('emotion', data_train['emotion'])
        datable('situation', data_train['situation'])
        datable('concepts', data_train['concepts'])
        datable('sample_concepts', data_train['sample_concepts'])
        datable('vads', data_train['vads'])
        datable('vad', data_train['vad'])
        datable('target_vad', data_train['target_vad'])
        datable('target_vads', data_train['target_vads'])
        # datable('train', data_train)
        return datable    

    def _read_train(self, data_key, path=None):
        return self._read(data_key)

    def _read_dev(self, data_key, path=None):
        return self._read(data_key)

    def _read_test(self, data_key, path=None):
        return self._read(data_key)

    def read_all(self):
        return self._read_train(data_key='train'), self._read_dev(data_key='val'), self._read_test(data_key='test')
        
    def read_vocab(self):
        with open(self.datapath, 'r') as f:
            data_train, data_val, data_test, vocab = json.load(f)
        return vocab

   

if __name__ == "__main__":
    reader = KempReader(raw_data_path="/data/zhaojingxuan/zjxcode/CogAgent/datapath/kemp/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    print("读取数据完成")