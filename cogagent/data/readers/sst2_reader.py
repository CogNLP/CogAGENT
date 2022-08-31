import os
from cogagent.data.readers.base_reader import BaseReader
from cogagent.data.datable import DataTable
from cogagent.utils.vocab_utils import Vocabulary


# from cogagent.utils.download_utils import Downloader


class Sst2Reader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        # downloader = Downloader()
        # downloader.download_sst2_raw_data(raw_data_path)
        self.train_file = 'train.tsv'
        self.dev_file = 'dev.tsv'
        self.test_file = 'test.tsv'
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)
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

    def _read_train(self, path=None):
        return self._read(path)

    def _read_dev(self, path=None):
        return self._read(path)

    def _read_test(self, path=None):
        print("Reading data...")
        datable = DataTable()
        with open(path) as file:
            lines = file.readlines()
        header = lines[0]
        contents = lines[1:]
        for line in contents:
            index, sentence = line.strip().split("\t")
            datable("sentence", sentence)
        return datable

    def read_all(self):
        return self._read_train(self.train_path), self._read_dev(self.dev_path), self._read_test(self.test_path)

    def read_vocab(self):
        self.label_vocab.create()
        return {"label_vocab": self.label_vocab}


if __name__ == "__main__":
    reader = Sst2Reader(raw_data_path="/data/mentianyi/code/CogAGENT/datapath/text_classification/SST_2/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    print("end")
