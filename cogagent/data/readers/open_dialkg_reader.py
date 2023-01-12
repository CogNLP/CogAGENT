import os
from cogagent.data.readers.base_reader import BaseReader
from cogagent.data.datable import DataTable
from cogagent.utils.vocab_utils import Vocabulary
import json
from tqdm import tqdm
from collections import Counter
from cogagent.data.processors.open_dialkg_processors.graph import NER
from cogagent.data.processors.open_dialkg_processors.kge import KnowledgeGraphEmbedding

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

    def read_vocab(self):

        # ner construction
        graph_file = os.path.join(self.raw_data_path,"opendialkg_triples.txt")
        if not os.path.exists(graph_file):
            graph_file = os.path.join(self.raw_data_path,"original_data/opendialkg_triples.txt")
            if not os.path.exists(graph_file):
                raise ValueError("Graph File {} could not be found in {}!".format("opendialkg_triples.txt",raw_data_path))
        self.ner = NER(self.raw_data_path,graph_file=graph_file)

        # kge construction
        self.kge = KnowledgeGraphEmbedding(os.path.join(self.raw_data_path,"graph_embedding"),self.train_path,self.dev_path,self.test_path)
        return {
            "kge":self.kge,
            "ner":self.ner,
        }

if __name__ == "__main__":
    from cogagent.utils.log_utils import init_logger
    logger = init_logger()
    reader = OpenDialKGReader(raw_data_path="/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/OpenDialKG/raw_data")
    train_data,dev_data,test_data = reader.read_all()
    vocab = reader.read_vocab()

    # from cogagent import save_pickle
    # save_pickle(train_data,"/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/OpenDialKG/raw_data/train_data.pkl")