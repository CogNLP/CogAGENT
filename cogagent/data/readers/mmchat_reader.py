import os
from cogagent.data.readers.base_reader import BaseReader
from cogagent.data.datable import DataTable
from tqdm import tqdm
from datasets import load_dataset
import json
from transformers import pipeline


class MMchatReader(BaseReader):

    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        self.image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
        self.raw_img_path = "/data/mentianyi/code/CogNLP/datapath/mm_dialog/mmchat/image"
        self.dataset = load_dataset("silver/mmchat", "mmchat_hf")
        self.img_index_data_path = os.path.join(self.raw_data_path, "weibo_img_index.json")
        self.url_data_path = os.path.join(self.raw_data_path, "weibo_img_expanded_url.json")
        self.url_unique_weiboid_data_path = os.path.join(self.raw_data_path,
                                                         "weibo_img_expanded_url_unique_weiboid.json")

        img_index_file = open(self.img_index_data_path, 'r', encoding='utf-8')
        self.img_index = {}
        for line in img_index_file.readlines():
            dic = json.loads(line)
            processed_path = []
            for item in dic['weibo_img_path'].split(";"):
                processed_path.append(os.path.join(self.raw_img_path, item))
            self.img_index[dic["weiboid"]] = processed_path

        url_file = open(self.url_data_path, 'r', encoding='utf-8')
        self.img2jpgid = {}
        for line in url_file.readlines():
            dic = json.loads(line)
            weibo_img_list = dic["weibo_img"].split(";")
            for item in weibo_img_list:
                self.img2jpgid[item] = self.img_index[dic['weiboid']]
        self.split = [98, 1, 1]

    def read_all(self):
        train_datable = DataTable()
        dev_datable = DataTable()
        test_datable = DataTable()
        print("Reading data...")
        for index, line in enumerate(tqdm(self.dataset["train"])):
            if index < len(self.dataset["train"]) * (self.split[0] / sum(self.split)):
                train_datable("dialog", line["dialog"])
                train_datable("weibo_content", line["weibo_content"])
                train_datable("imgs", self.img2jpgid[line["imgs"][0]])
            elif index < len(self.dataset["train"]) * ((self.split[0] + self.split[1]) / sum(self.split)):
                dev_datable("dialog", line["dialog"])
                dev_datable("weibo_content", line["weibo_content"])
                dev_datable("imgs", self.img2jpgid[line["imgs"][0]])
            else:
                test_datable("dialog", line["dialog"])
                test_datable("weibo_content", line["weibo_content"])
                test_datable("imgs", self.img2jpgid[line["imgs"][0]])
        return train_datable, dev_datable, test_datable


if __name__ == "__main__":
    reader = MMchatReader(
        raw_data_path="/data/mentianyi/code/CogNLP/datapath/mm_dialog/mmchat/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    print("end")
