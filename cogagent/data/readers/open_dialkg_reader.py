import os
from cogagent.data.readers.base_reader import BaseReader
from cogagent.data.datable import DataTable
from cogagent.utils.vocab_utils import Vocabulary
import json
from tqdm import tqdm
from collections import Counter

class OpenDialKGReader(BaseReader):
    def __init__(self, raw_data_path,test_mode='seen'):
        super().__init__()
        if test_mode not in ['seen','unseen']:
            assert ValueError("Test mode must be seen or unseen but got {}!".format(test_mode))
        self.raw_data_path = raw_data_path
        self.train_file = 'train_nph_data.json'
        self.dev_file = 'dev_nph_data.json'
        self.test_file = 'test_nph_data.json'

        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)

        self.max_history = 3

    def _read(self,path=None):
        print("Reading data...")
        datable = DataTable()
        with open(path) as file:
            raw_lines = [json.loads(line) for line in file]

        for line in tqdm(raw_lines):
            line["history"] = line["history"][-self.max_history : ]
            for key,value in line.items():
                datable(key,value)

        return datable

    def _read_train(self):
        return self._read(self.train_path)

    def _read_dev(self):
        return self._read(self.dev_path)

    def _read_test(self):
        return self._read(self.test_path)

    def read_all(self):
        return [self._read_train(),self._read_dev(),self._read_test()]

if __name__ == "__main__":
    reader = OpenDialKGReader(raw_data_path="/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/OpenDialKG/raw_data")
    train_data,dev_data,test_data = reader.read_all()
    # from cogagent import save_pickle
    # save_pickle(train_data,"/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/OpenDialKG/raw_data/train_data.pkl")