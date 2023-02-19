import os
from cogagent.data.readers.base_reader import BaseReader
from cogagent.data.datable import DataTable
from cogagent.utils.vocab_utils import Vocabulary
import json

# from cogagent.utils.download_utils import Downloader


class DiaSafetyReader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        self.train_file = 'train.json'
        self.dev_file = 'val.json'
        self.test_file = 'test.json'
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)
        self.label_vocab = Vocabulary()

    def _read(self, path=None):
        print("Reading data...")
        datable = DataTable()
        with open(path, 'r') as f:
            lines = json.load(f)
        for line in lines:
            for key in ["context","response","category","label"]:
                datable(key,line[key])
            self.label_vocab.add(line["label"])
        return datable

    def _read_train(self, path=None):
        return self._read(path)

    def _read_dev(self, path=None):
        return self._read(path)

    def _read_test(self, path=None):
        return self._read(path)

    def read_all(self):
        return self._read_train(self.train_path), self._read_dev(self.dev_path), self._read_test(self.test_path)

    def read_vocab(self):
        self.label_vocab.create()
        return {"label_vocab": self.label_vocab}


if __name__ == "__main__":
    reader = DiaSafetyReader(raw_data_path="/data/hongbang/CogAGENT/datapath/dialogue_safety/DiaSafety/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    print("end")
