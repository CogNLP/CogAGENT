import scipy as sp
from cogagent.data.readers.base_reader import BaseReader
from cogagent.data.datable import DataTable
from cogagent.utils.vocab_utils import Vocabulary
import os

import json
import linecache
import faiss
import numpy as np

import pytrec_eval

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

# 数据读取

class MMCoQAReader(BaseReader):
    def __init__(self, raw_data_path):
        super().__init__()
        self.proj_size = 128
        self.label_vocab = Vocabulary()
        self.raw_data_path = raw_data_path
        # downloader = Downloader()
        # downloader.download_sst2_raw_data(raw_data_path)
        self.passages_file = 'multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl'
        self.tables_file = 'multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl'
        self.images_file = 'multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl'
        self.passage_rep = '/data/yangcheng/CogAgent/datapath/MMCoQA_data/passage_rep/dev_blocks.txt'
        self.train_file = 'MMCoQA_train.txt'
        self.dev_file = 'MMCoQA_dev.txt'
        self.test_file = 'MMCoQA_test.txt'
        self.qrels_file = 'qrels.txt'
        self.qrels_path = os.path.join(raw_data_path, self.qrels_file)
        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)
        self.passages_file_path = os.path.join(raw_data_path, self.passages_file)
        self.images_file_path = os.path.join(raw_data_path, self.images_file)
        self.tables_file_path = os.path.join(raw_data_path, self.tables_file)
        # self.label_vocab = Vocabulary()
        self.itemid_modalities = []  # load passages to passages_dict
        # load passages to passages_dict
        self.passages_dict = {}
        with open(self.passages_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line.strip())
                self.passages_dict[line['id']] = line['text']
                self.itemid_modalities.append('text')

        #load tables to tables_dict
        self.tables_dict = {}
        self.raw_tables_dict = {}
        with open(self.tables_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line.strip())
                self.raw_tables_dict[line['id']] = line
                table_context = ''
                for row_data in line['table']["table_rows"]:
                    for cell in row_data:
                        table_context = table_context + " " + cell['text']
                self.tables_dict[line['id']] = table_context
                self.itemid_modalities.append('table')


        # load images to images_dict
        self.images_dict = {}
        with open(self.images_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line.strip())
                self.images_dict[line['id']] = "/data/yangcheng/CogAgent/datapath/MMCoQA_data/data/final_dataset_images/" + line['path']
                self.itemid_modalities.append('image')

        #  get image-answer-set
        self.image_answers_set = set()
        with open(self.train_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                self.image_answers_set.add(json.loads(line.strip())['answer'][0]['answer'])
        self.image_answers_str = ''
        for s in self.image_answers_set:
            self.image_answers_str = self.image_answers_str + " " + str(s)

        self.images_titles = {}
        with open(self.images_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line.strip())
                self.images_titles[line['id']] = line['title'] + " " + self.image_answers_str

        self.item_ids, self.item_reps = [], []
        with open(self.passage_rep) as fin:
            for line in tqdm(fin):
                dic = json.loads(line.strip())
                self.item_ids.append(dic['id'])
                self.item_reps.append(dic['rep'])
                # if(dic['id'] not in passages_dict):
                #     item_reps.append(dic['rep'])
        self.item_reps = np.asarray(self.item_reps, dtype='float32')  # retrieve_need
        self.item_ids = np.asarray(self.item_ids)  # retrieve_need

        self.faiss_res = faiss.StandardGpuResources()
        self.index = faiss.IndexFlatIP(self.proj_size)
        self.index.add(self.item_reps)
        # gpu_index = faiss.index_cpu_to_gpu(faiss_res, 0, index)
        self.gpu_index = self.index  # retrieve_need

        # qrels evaluate时候需要
        with open(self.qrels_path) as handle:
            self.qrels = json.load(handle)

        # 以下步骤为了得到 qrels_sparse_matrix
        self.item_id_to_idx = {}  # retrieve_need
        for i, pid in enumerate(self.item_ids):
            self.item_id_to_idx[pid] = i
        self.qrels_data, self.qrels_row_idx, self.qrels_col_idx = [], [], []
        self.qid_to_idx = {}  # retrieve_need
        for i, (qid, v) in enumerate(self.qrels.items()):
            self.qid_to_idx[qid] = i
            for pid in v.keys():
                self.qrels_data.append(1)
                self.qrels_row_idx.append(i)
                self.qrels_col_idx.append(self.item_id_to_idx[pid])
        self.qrels_data.append(0)
        self.qrels_row_idx.append(5752)
        self.qrels_col_idx.append(285384)
        self.qrels_sparse_matrix = sp.sparse.csr_matrix(  # retrieve_need
            (self.qrels_data, (self.qrels_row_idx, self.qrels_col_idx)))

        self.evaluator = pytrec_eval.RelevanceEvaluator(self.qrels, {'ndcg', 'set_recall'})


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
        print("Reading train data...")
        datable = DataTable()

        data = []
        with open(self.train_path, 'r') as f:
            lines = f.readlines()
        for i in range(len(lines)):
        # for i in range(500):
            train_data = linecache.getline(self.train_path, i + 1)
            datable("train_data", train_data)
        return datable

    def _read_dev(self, path=None):
        print("Reading dev data...")
        datable = DataTable()
        with open(self.dev_path, 'r') as f:
            lines = f.readlines()
        for i in range(len(lines)):
            # for i in range(500):
            dev_data = linecache.getline(self.dev_path, i + 1)
            datable("dev_data", dev_data)
        return datable

    def _read_test(self, path=None):
        print("Reading data...")
        datable = DataTable()
        with open(self.test_path, 'r') as f:
            lines = f.readlines()
        for i in range(len(lines)):
            # for i in range(500):
            test_data = linecache.getline(self.test_path, i + 1)
            datable("train_data", test_data)
        return datable

    def read_all(self):
        return self._read_train(self.train_path), self._read_dev(self.dev_path), self._read_test(self.test_path)

    def read_vocab(self):
        self.label_vocab.create()
        return {"label_vocab": self.label_vocab}

    def read_addition(self):
        batch = {}
        batch["qid_to_idx"] = self.qid_to_idx
        batch["item_ids"] = self.item_ids
        batch["item_id_to_idx"] = self.item_id_to_idx
        batch["item_reps"] = self.item_reps
        batch["qrels"] = self.qrels
        batch["qrels_sparse_matrix"] = self.qrels_sparse_matrix
        batch["gpu_index"] = self.gpu_index
        batch["itemid_modalities"] = self.itemid_modalities
        batch["passages_dict"] = self.passages_dict
        batch["tables_dict"] = self.tables_dict
        batch["images_dict"] = self.images_dict
        batch["images_titles"] = self.images_titles
        batch["raw_tables_dict"] = self.raw_tables_dict
        return batch





if __name__ == "__main__":
    reader = MMCoQAReader(raw_data_path="/data/yangcheng/CogAgent/datapath/MMCoQA_data/data/")
    train_data, dev_data, test_data = reader.read_all()
    train_batch = reader.read_addition()
    vocab = reader.read_vocab()
    print("end")
