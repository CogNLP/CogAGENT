from cogagent.toolkits.base_toolkit import BaseToolkit
from cogagent.utils import load_model
from cogagent.models.led_dialog import LedDialog
import torch
import numpy as np
import json
import re
from nltk.tokenize import word_tokenize
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}

    def add_word(self, word, mode):
        if word not in self.word2idx and mode in ("train"):
            idx = len(self.idx2word)
            self.idx2word[idx] = word
            self.word2idx[word] = idx
            return word
        elif word not in self.word2idx and mode != "train":
            return "<unk>"
        else:
            return word

    def __len__(self):
        return len(self.idx2word)


class EmbodiedToolkit(BaseToolkit):
    """
    """

    def __init__(self,
                 model_path=None,
                 embedding_dir=None,
                 raw_data_path=None,
                 device="cpu"):
        super().__init__()
        self.model_path = model_path
        self.embedding_dir = embedding_dir
        self.raw_data_path = raw_data_path
        self.device = device

        self.max_length = 0

        self.vocab = Vocabulary()
        word_embeddings_file = "word_embeddings/"
        word_embeddings_path = os.path.join(raw_data_path, word_embeddings_file)
        self.vocab.word2idx = json.load(open(word_embeddings_path + "word2idx.json"))
        self.vocab.idx2word = json.load(open(word_embeddings_path + "idx2word.json"))
        self.pix2meshdistance_file = 'floorplans/pix2meshDistance.json'
        self.pix2meshdistance_path = os.path.join(raw_data_path, self.pix2meshdistance_file)
        self.mesh2meters = json.load(open(self.pix2meshdistance_path))
        self.floorplans_file = 'floorplans/'
        self.floorplans_path = os.path.join(raw_data_path, self.floorplans_file)
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.555],
                    std=[0.229, 0.224, 0.225, 0.222],
                ),
            ]
        )

        addition = {}
        addition["node2pix_path"] = os.path.join(raw_data_path, 'floorplans/allScans_Node2pix.json')
        addition["geodistance_nodes_path"] = os.path.join(raw_data_path, 'geodistance_nodes.json')
        addition["vocab"] = self.vocab
        self.model = LedDialog(addition=addition,
                               embedding_dir=self.embedding_dir)
        self.model = load_model(self.model, self.model_path)
        self.model = self.model.to(self.device)

    def _add_tokens(self, message_arr):
        new_dialog = ""
        for enum, message in enumerate(message_arr):
            if enum % 2 == 0:
                new_dialog += "SOLM " + message + " EOLM "
            else:
                new_dialog += "SOOM " + message + " EOOM "
        return new_dialog

    def _build_pretrained_vocab(self, texts):
        ids = []
        seq_lengths = []
        for text in texts:
            text = re.sub(r"\.\.+", ". ", text)
            line_ids = []
            words = word_tokenize(text.lower())
            for word in words:
                if word in self.vocab.word2idx:
                    line_ids.append(self.vocab.word2idx[word])
            ids.append(line_ids)
            seq_lengths.append(len(line_ids))
            self.max_length = max(self.max_length, len(ids))
        text_ids = np.array([row + [0] * (self.max_length - len(row)) for row in ids])
        return text_ids, seq_lengths

    def _process_one(self, raw_dict):
        processed_dict = {}
        dialog = self._add_tokens(raw_dict["dialog_list"])
        text, seq_length = self._build_pretrained_vocab([dialog])

        all_maps = torch.zeros(
            5,
            3,
            455,
            780,
        )

        sn = raw_dict["house_name"]
        floors = self.mesh2meters[sn].keys()
        img_list = []
        for enum, f in enumerate(floors):
            img = Image.open(
                "{}floor_{}/{}_{}.png".format(self.floorplans_path, f, sn, f)
            )
            img = img.resize((780, 455))
            img_list.append(img)
            all_maps[enum, :, :, :] = self.preprocess(img)[:3, :, :]
        processed_dict["text"] = torch.LongTensor(text).to(self.device)
        processed_dict["seq_length"] = torch.tensor(seq_length).to(self.device)
        processed_dict["maps"] = torch.tensor(all_maps).unsqueeze(0).to(self.device)
        processed_dict["img_list"] = [img_list]
        return processed_dict

    def infer_one(self,
                  house_name=None,
                  raw_dict=None):
        floors = self.mesh2meters[house_name].keys()
        img_list = []
        for enum, f in enumerate(floors):
            img = Image.open(
                "{}floor_{}/{}_{}.png".format(self.floorplans_path, f, house_name, f)
            )
            plt.imshow(img)
            plt.show()
        while True:
            try:
                raw_dict = {}
                raw_dict["dialog_list"] = []
                raw_dict["dialog_list"].append('What do you see?')
                raw_dict["house_name"] = house_name
                # desc_sentence = input("Your description:")
                desc_sentence = "I'm next to the bed. The floor is yellow"
                raw_dict["dialog_list"].append(desc_sentence)
                processed_dict = self._process_one(raw_dict)

                preds, result_list = self.model.predict(batch=processed_dict)
            except KeyboardInterrupt:
                print("聊天结束.")
                break
        return None


if __name__ == "__main__":
    embodiedtoolkit = EmbodiedToolkit(
        model_path="/data/mentianyi/code/CogNLP/datapath/embodied_dialog/way/experimental_result/final--2023-01-21--22-30-01.86/model/checkpoint-48000/models.pt",
        raw_data_path="/data/mentianyi/code/CogNLP/datapath/embodied_dialog/way/raw_data",
        embedding_dir="/data/mentianyi/code/CogNLP/datapath/embodied_dialog/way/raw_data/word_embeddings/",
        device="cuda:0")
    infer_dict = embodiedtoolkit.infer_one(house_name='5q7pvUzZiYa')
    print(infer_dict)
    print("end")  # 300
