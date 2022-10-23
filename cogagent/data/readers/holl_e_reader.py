import os
from cogagent.data.readers.base_reader import BaseReader
from cogagent.data.datable import DataTable
from cogagent.utils.vocab_utils import Vocabulary
import json
from tqdm import tqdm
from collections import Counter

class HollEReader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        self.train_file = 'train_data.json'
        self.dev_file = 'dev_data.json'
        self.test_file = 'test_data.json'
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)

        self._vocab_size = 16000
        self._invalid_vocab_times = 0

        self.raw_vocab_list = []
        self.ext_vocab =  ["<pad>", "<unk>", "<go>", "<eos>"]
        self.vocab_list = None
        self.valid_vocab_len = None
        self.valid_vocab_set = None

    def _read(self, path=None):
        print("Reading data...")
        datable = DataTable()
        with open(path) as file:
            origin =json.load(file)

        for dialog in tqdm(origin):
            posts = []
            for single_post in dialog['post']:
                single_post = single_post.split()[:]
                self.raw_vocab_list.extend(single_post * 10)
                posts.append(single_post)
            datable("post",posts)

            resps = []
            for single_resp in dialog['resp']:
                single_resp = single_resp.split()[:] if isinstance(single_resp, str) else single_resp[0].split()[:]
                self.raw_vocab_list.extend(single_resp * 10)
                resps.append(single_resp)
            datable("resp", resps)

            knows = []
            for single_know in dialog['know']:
                single_know = [each.split()[:] for each in single_know]
                for each in single_know:
                    self.raw_vocab_list.extend(each)
                knows.append(single_know)
            datable("wiki",knows)

            datable("atten",[each + 1 for each in dialog['atten']])

        return datable

    def _read_train(self, path=None):
        train_datable = self._read(path)
        vocab = Counter(self.raw_vocab_list).most_common()
        left_vocab = list(map(lambda x: x[0], vocab[:self._vocab_size]))
        self.vocab_list = self.ext_vocab + left_vocab
        self.valid_vocab_len = len(self.vocab_list)
        self.valid_vocab_set = set(self.vocab_list)
        return train_datable

    def _read_dev(self, path=None):
        return self._read(path)

    def _read_test(self, path=None):
        test_datable = self._read(path)
        vocab = Counter(self.raw_vocab_list).most_common()
        left_vocab = list(filter(lambda x: x[1] >= self._invalid_vocab_times and x[0] not in self.valid_vocab_set, vocab))
        left_vocab = list(map(lambda x: x[0], left_vocab))
        self.vocab_list.extend(left_vocab)
        return test_datable

    def read_all(self):
        return self._read_train(self.train_path), self._read_dev(self.dev_path), self._read_test(self.test_path)

    def read_vocab(self):
        vocab = Vocabulary()
        vocab.add_dict({w: i for i, w in enumerate(self.vocab_list)})
        vocab.create()
        return {"word_vocab": vocab}


if __name__ == "__main__":
    reader = HollEReader(raw_data_path="/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/holl_e/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()
    print("end")
