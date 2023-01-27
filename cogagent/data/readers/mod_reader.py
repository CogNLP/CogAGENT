import os
from cogagent.data.readers.base_reader import BaseReader
from cogagent.data.datable import DataTable
from tqdm import tqdm
import json


class ModReader(BaseReader):

    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        self.train_file = 'train_pair.json'
        self.dev_file = 'validation_pair.json'
        self.id2img_file = "id2img.json"
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.id2img_path = os.path.join(raw_data_path, self.id2img_file)

    def _read(self, datatype=None):
        datable = DataTable()
        print("Reading data...")
        with open(datatype, encoding='utf-8') as f:
            lines = json.load(f)
        for line in tqdm(lines):
            selected_dialog = line["dialog"]
            sent = ''
            img_id = None
            for i, d in enumerate(selected_dialog):
                if i == len(selected_dialog) - 1:
                    s = d.get('text', None)
                    img_id = d.get('img_id', None)
                    img_word = d.get('img_label', None)
                    neg_img_id = d.get('neg_img_id', None)
                    neg_img_word = d.get('neg_img_label', None)
                    emotion_id = d.get('emotion_id', None)

                else:
                    s = d['text']
                sent += s
            datable("sent", sent)
            datable("img_id", img_id)
            datable("img_word", img_word)
            datable("neg_img_id", neg_img_id)
            datable("neg_img_word", neg_img_word)
            datable("emotion_id", emotion_id)

        return datable

    def _read_train(self):
        return self._read(datatype=self.train_path)

    def _read_dev(self):
        return self._read(datatype=self.dev_path)

    def _read_test(self):
        return None

    def read_all(self):
        return self._read_train(), self._read_dev(), self._read_test()

    def read_addition(self):
        addition_dict = {}
        addition_dict["id2img"] = {}
        print("Reading data...")
        with open(self.id2img_path, encoding='utf-8') as f:
            id2img = json.load(f)
            id2imgpath = {}
            for id, img in tqdm(id2img.items()):
                id = int(id)
                id2imgpath[id] = img
                addition_dict["id2imgpath"] = id2imgpath
        return addition_dict


if __name__ == "__main__":
    reader = ModReader(raw_data_path="/data/mentianyi/code/CogNLP/datapath/mm_dialog/mod/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    addition = reader.read_addition()
    print("end")
