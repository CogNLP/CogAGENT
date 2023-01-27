import os
from cogagent.data.readers.base_reader import BaseReader
from cogagent.data.datable import DataTable
import json
import networkx as nx
import copy
import re
import numpy as np
from nltk.tokenize import word_tokenize


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


class WayReader(BaseReader):

    def __init__(self, raw_data_path):
        super().__init__()
        self.raw_data_path = raw_data_path
        self.connectivity_file = 'connectivity/'
        self.floorplans_file = 'floorplans/'
        self.way_splits_file = "way_splits/"
        self.word_embeddings_file = "word_embeddings/"
        self.scans_file = 'connectivity/scans.txt'
        self.pix2meshdistance_file = 'floorplans/pix2meshDistance.json'
        self.node2pix_file = 'floorplans/allScans_Node2pix.json'
        self.geodistance_nodes_file = 'geodistance_nodes.json'

        self.train_file = 'train_expanded_data.json'
        self.seen_valid_file = 'valSeen_data.json'
        self.unseen_valid_file = 'valUnseen_data.json'
        self.test_file = 'test_data.json'

        self.connectivity_path = os.path.join(raw_data_path, self.connectivity_file)
        self.floorplans_path = os.path.join(raw_data_path, self.floorplans_file)
        self.scans_path = os.path.join(raw_data_path, self.scans_file)
        self.pix2meshdistance_path = os.path.join(raw_data_path, self.pix2meshdistance_file)
        self.word_embeddings_path = os.path.join(raw_data_path, self.word_embeddings_file)
        self.node2pix_path = os.path.join(raw_data_path, self.node2pix_file)
        self.geodistance_nodes_path = os.path.join(raw_data_path, self.geodistance_nodes_file)

        self.train_path = os.path.join(raw_data_path, self.way_splits_file, self.train_file)
        self.seen_valid_path = os.path.join(raw_data_path, self.way_splits_file, self.seen_valid_file)
        self.unseen_valid_path = os.path.join(raw_data_path, self.way_splits_file, self.unseen_valid_file)
        self.test_path = os.path.join(raw_data_path, self.way_splits_file, self.test_file)

        self.mesh2meters = json.load(open(self.pix2meshdistance_path))
        self.vocab = Vocabulary()
        self.vocab.word2idx = json.load(open(self.word_embeddings_path + "word2idx.json"))
        self.vocab.idx2word = json.load(open(self.word_embeddings_path + "idx2word.json"))
        self.max_length = 0

    def _distance(self, pose1, pose2):
        """Euclidean distance between two graph poses"""
        return (
                       (pose1["pose"][3] - pose2["pose"][3]) ** 2
                       + (pose1["pose"][7] - pose2["pose"][7]) ** 2
                       + (pose1["pose"][11] - pose2["pose"][11]) ** 2
               ) ** 0.5

    def _open_graph(self, connectDir, scan_id):
        """Build a graph from a connectivity json file"""
        infile = "%s%s_connectivity.json" % (connectDir, scan_id)
        G = nx.Graph()
        with open(infile) as f:
            data = json.load(f)
            for i, item in enumerate(data):
                if item["included"]:
                    for j, conn in enumerate(item["unobstructed"]):
                        if conn and data[j]["included"]:
                            assert data[j]["unobstructed"][i], "Graph should be undirected"
                            G.add_edge(
                                item["image_id"],
                                data[j]["image_id"],
                                weight=self._distance(item, data[j]),
                            )
        return G

    def _load_locations(self, data, mode):
        if "test" in mode:
            return [[0, 0] for _ in data], ["" for _ in data]

        x = [
            [
                data_obj["finalLocation"]["pixel_coord"][1],
                data_obj["finalLocation"]["pixel_coord"][0],
            ]
            for data_obj in data
        ]

        y = [data_obj["finalLocation"]["viewPoint"] for data_obj in data]

        return x, y

    def _load_image_paths(self, data, mode):
        episode_ids, scan_names, levels, mesh_conversions, dialogs = [], [], [], [], []
        for data_obj in data:
            episode_ids.append(data_obj["episodeId"])
            scan_names.append(data_obj["scanName"])
            # 加上token，使对话历史连成一句话
            dialogs.append(self._add_tokens(data_obj["dialogArray"]))
            level = 0
            if mode != "test":
                level = str(data_obj["finalLocation"]["floor"])
            levels.append(level)
            mesh_conversions.append(
                self.mesh2meters[data_obj["scanName"]][str(level)]["threeMeterRadius"]
                / 3.0
            )
        return episode_ids, scan_names, levels, mesh_conversions, dialogs

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
            self.max_length = max(self.max_length, len(words))
            for word in words:
                line_ids.append(self.vocab.word2idx[word])
            ids.append(line_ids)
            seq_lengths.append(len(words))
        text_ids = np.array([row + [0] * (self.max_length - len(row)) for row in ids])
        return text_ids, seq_lengths

    def _read(self, mode=None, path=None):
        datable = DataTable()
        print("Reading data...")
        data = json.load(open(path))
        locations, viewPoint_location = self._load_locations(data, mode)
        (
            episode_ids,
            scan_names,
            levels,
            mesh_conversions,
            dialogs,
        ) = self._load_image_paths(data, mode)
        texts = copy.deepcopy(dialogs)
        texts, seq_lengths = self._build_pretrained_vocab(texts)

        data_num = len(scan_names)
        for i in range(data_num):
            datable("texts", texts[i])
            datable("seq_lengths", seq_lengths[i])
            datable("mesh_conversions", mesh_conversions[i])
            datable("locations", locations[i])
            datable("viewPoint_location", viewPoint_location[i])
            datable("dialogs", dialogs[i])
            datable("scan_names", scan_names[i])
            datable("levels", levels[i])
            datable("episode_ids", episode_ids[i])
        return datable

    def _read_train(self):
        return self._read(mode='train', path=self.train_path)

    def _read_seen_dev(self):
        return self._read(mode='valSeen', path=self.seen_valid_path)

    def _read_unseen_dev(self):
        return self._read(mode='valUnseen', path=self.unseen_valid_path)

    def _read_test(self):
        return self._read(mode="test", path=self.test_path)

    def read_all(self):
        return self._read_train(), \
               self._read_seen_dev(), \
               self._read_unseen_dev(), \
               self._read_test()

    def read_addition(self):
        addition_dict = {}
        addition_dict["scan_graphs"] = {}
        addition_dict["vocab"] = None

        print("Reading data...")
        scan_graphs = {}
        with open(self.scans_path) as file:
            scans_list = file.readlines()
            for scan_id in scans_list:
                scan_id = scan_id.strip()
                scan_graphs[scan_id] = self._open_graph(self.connectivity_path, scan_id)

        addition_dict["scan_graphs"] = scan_graphs
        addition_dict["vocab"] = self.vocab
        addition_dict["mesh2meters"] = self.mesh2meters
        addition_dict["floorplans_path"] = self.floorplans_path
        addition_dict["node2pix_path"] = self.node2pix_path
        addition_dict["geodistance_nodes_path"] = self.geodistance_nodes_path
        return addition_dict


if __name__ == "__main__":
    reader = WayReader(raw_data_path="/data/mentianyi/code/CogNLP/datapath/embodied_dialog/way/raw_data")
    train_data, dev_seen_data, dev_unseen_data, test_data = reader.read_all()
    addition = reader.read_addition()
    print("end")
