import os
from cogagent.data.readers.base_reader import BaseReader
from cogagent.data.datable import DataTable
from tqdm import tqdm


class ChineseGossipDialogReader(BaseReader):

    def __init__(self, raw_data_path, data_num="100w"):
        super().__init__()
        self.raw_data_path = raw_data_path
        self.data_num = data_num
        if self.data_num == '50w':
            self.train_file = 'train_50w.txt'
        if self.data_num == '100w':
            self.train_file = 'train_100w.txt'
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.split = [998, 1, 1]

    def _read(self, datatype=None):
        datable = DataTable()
        print("Reading data...")
        with open(self.train_path, 'rb') as file:
            data = file.read().decode("utf-8")
        if "\r\n" in data:
            data = data.split("\r\n\r\n")
        else:
            data = data.split("\n\n")

        for index, dialogue in enumerate(tqdm(data)):
            if "\r\n" in dialogue:
                utterances = dialogue.split("\r\n")
            else:
                utterances = dialogue.split("\n")
            datable("utterances", utterances)

        return datable

    def read_all(self):
        data = self._read(datatype=self.train_path)
        data_num = len(data)
        train_start_index = 0
        train_end_index = int((self.split[0] / sum(self.split)) * data_num)
        dev_start_index = train_end_index
        dev_end_index = int(((self.split[0] + self.split[1]) / sum(self.split)) * data_num)
        test_start_index = dev_end_index
        test_end_index = int(((self.split[0] + self.split[1] + self.split[2]) / sum(self.split)) * data_num) + 1
        train_data = DataTable()
        dev_data = DataTable()
        test_data = DataTable()
        for header in data.headers:
            train_data[header] = data[header][train_start_index:train_end_index]
            dev_data[header] = data[header][dev_start_index:dev_end_index]
            test_data[header] = data[header][test_start_index:test_end_index]
        return train_data, dev_data, test_data


if __name__ == "__main__":
    reader = ChineseGossipDialogReader(
        raw_data_path="/data/mentianyi/code/CogNLP/datapath/gossip_dialog/chinese_gossip_dialog/raw_data",
        data_num="100w")
    train_data, dev_data, test_data = reader.read_all()
    print("end")
