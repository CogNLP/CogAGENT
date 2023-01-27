import os
from cogagent.data.readers.base_reader import BaseReader
from cogagent.data.datable import DataTable
from tqdm import tqdm


class Convai2Reader(BaseReader):


    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        self.train_file = 'train_self_original.txt'
        self.dev_file = 'valid_self_original.txt'
        self.positive_file = 'nli_positive.tsv'
        self.negative_file = 'nli_negative.tsv'
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.positive_path = os.path.join(raw_data_path, self.positive_file)
        self.negative_path = os.path.join(raw_data_path, self.negative_file)

    def _read(self, datatype=None):
        datable = DataTable()
        print("Reading data...")
        data_num = sum([1 for i in open(datatype, "r")])
        with open(datatype, "r", encoding="utf-8") as file:
            pre_st, st = 'dia', 'dia'
            for line in tqdm(file, total=data_num):
                line = line.strip()
                if 'your persona:' in line:
                    pre_st = st
                    st = 'per'
                else:
                    pre_st = st
                    st = 'dia'
                if pre_st == 'dia' and st == 'per':
                    per_group = ''
                if st == 'per':
                    per_group += (line[16:] + ' ')
                elif st == 'dia':
                    datable("persona", per_group)
                    line = line[line.find(' '):]
                    datable("query", line.split('\t')[0])
                    datable("response", line.split('\t')[1])
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
        # pre长，hyp短
        addition_dict = {}
        addition_dict["pre_positive"] = {}
        addition_dict["hyp_positive"] = {}
        addition_dict["pre_negative"] = {}
        addition_dict["hyp_negative"] = {}
        print("Reading data...")
        pre_positive = []
        hyp_positive = []
        pre_negative = []
        hyp_negative = []

        positive_data_num = sum([1 for i in open(self.positive_path, "r")])
        with open(self.positive_path, encoding='utf-8') as file:
            for line in tqdm(file, total=positive_data_num):
                line = line.strip()
                sent_1, sent_2 = line.split('\t')[0], line.split('\t')[1]
                if len(sent_1.split(' ')) > len(sent_2.split(' ')):
                    pre, hyp = sent_1, sent_2
                else:
                    pre, hyp = sent_2, sent_1
                pre_positive.append(pre)
                hyp_positive.append(hyp)
        addition_dict["pre_positive"] = pre_positive
        addition_dict["hyp_positive"] = hyp_positive

        negative_data_num = sum([1 for i in open(self.negative_path, "r")])
        with open(self.negative_path, encoding='utf-8') as file:
            for line in tqdm(file, total=negative_data_num):
                line = line.strip()
                sent_1, sent_2 = line.split('\t')[0], line.split('\t')[1]
                if len(sent_1.split(' ')) > len(sent_2.split(' ')):
                    pre, hyp = sent_1, sent_2
                else:
                    pre, hyp = sent_2, sent_1
                pre_negative.append(pre)
                hyp_negative.append(hyp)
        addition_dict["pre_negative"] = pre_negative
        addition_dict["hyp_negative"] = hyp_negative

        return addition_dict


if __name__ == "__main__":
    reader = Convai2Reader(raw_data_path="/data/mentianyi/code/CogNLP/datapath/controllable_dialog/convai2/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    addition = reader.read_addition()
    print("end")
