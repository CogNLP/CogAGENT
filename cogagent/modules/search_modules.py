import csv
from tqdm import tqdm
import hydra
import torch
from cogagent.utils.log_utils import logger
from transformers import BertModel,BertConfig,BertTokenizer
import torch.nn as nn
from typing import Tuple, List
import transformers
from torch import Tensor as T
import numpy as np
from collections import OrderedDict
import os
import faiss
import pickle
from cogagent.utils.io_utils import load_pickle


class HFBertEncoder(BertModel):
    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.init_weights()

    @classmethod
    def init_encoder(
        cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.1, pretrained: bool = True, **kwargs
    ) -> BertModel:
        logger.info("Initializing HF BERT Encoder. cfg_name=%s", cfg_name)
        cfg = BertConfig.from_pretrained(cfg_name if cfg_name else "bert-base-uncased")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        if pretrained:
            return cls.from_pretrained(cfg_name, config=cfg, project_dim=projection_dim, **kwargs)
        else:
            return HFBertEncoder(cfg, project_dim=projection_dim)

    def forward(
        self,
        input_ids: T,
        token_type_ids: T,
        attention_mask: T,
        representation_token_pos=0,
    ) -> Tuple[T, ...]:

        out = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        # HF >4.0 version support
        if transformers.__version__.startswith("4") and isinstance(
            out,
            transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions,
        ):
            sequence_output = out.last_hidden_state
            pooled_output = None
            hidden_states = out.hidden_states

        elif self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = out
        else:
            hidden_states = None
            out = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            sequence_output, pooled_output = out

        if isinstance(representation_token_pos, int):
            pooled_output = sequence_output[:, representation_token_pos, :]
        else:  # treat as a tensor
            bsz = sequence_output.size(0)
            assert representation_token_pos.size(0) == bsz, "query bsz={} while representation_token_pos bsz={}".format(
                bsz, representation_token_pos.size(0)
            )
            pooled_output = torch.stack([sequence_output[i, representation_token_pos[i, 1], :] for i in range(bsz)])

        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states


    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size



class DenseIndexer(object):
    def __init__(self, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        self.index_id_to_db_id = []
        self.index = None

    def init_index(self, vector_sz: int):
        raise NotImplementedError

    def index_data(self, data: List[Tuple[object, np.array]]):
        raise NotImplementedError

    def get_index_name(self):
        raise NotImplementedError

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
        raise NotImplementedError

    def serialize(self, file: str):
        logger.info("Serializing index to %s", file)

        if os.path.isdir(file):
            index_file = os.path.join(file, "index.dpr")
            meta_file = os.path.join(file, "index_meta.dpr")
        else:
            index_file = file + ".index.dpr"
            meta_file = file + ".index_meta.dpr"

        faiss.write_index(self.index, index_file)
        with open(meta_file, mode="wb") as f:
            pickle.dump(self.index_id_to_db_id, f)

    def get_files(self, path: str):
        if os.path.isdir(path):
            index_file = os.path.join(path, "index.dpr")
            meta_file = os.path.join(path, "index_meta.dpr")
        else:
            index_file = path + ".{}.dpr".format(self.get_index_name())
            meta_file = path + ".{}_meta.dpr".format(self.get_index_name())
        return index_file, meta_file

    def index_exists(self, path: str):
        index_file, meta_file = self.get_files(path)
        return os.path.isfile(index_file) and os.path.isfile(meta_file)

    def deserialize(self, path: str):
        logger.info("Loading index from %s", path)
        index_file, meta_file = self.get_files(path)

        self.index = faiss.read_index(index_file)
        logger.info("Loaded index of type %s and size %d", type(self.index), self.index.ntotal)

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert (
            len(self.index_id_to_db_id) == self.index.ntotal
        ), "Deserialized index_id_to_db_id should match faiss index size"

    def _update_id_mapping(self, db_ids: List) -> int:
        self.index_id_to_db_id.extend(db_ids)
        return len(self.index_id_to_db_id)



class DenseFlatIndexer(DenseIndexer):
    def __init__(self, buffer_size: int = 50000):
        super(DenseFlatIndexer, self).__init__(buffer_size=buffer_size)

    def init_index(self, vector_sz: int):
        self.index = faiss.IndexFlatIP(vector_sz)

    def index_data(self, data: List[Tuple[object, np.array]]):
        n = len(data)
        # indexing in batches is beneficial for many faiss index types
        for i in range(0, n, self.buffer_size):
            db_ids = [t[0] for t in data[i : i + self.buffer_size]]
            vectors = [np.reshape(t[1], (1, -1)) for t in data[i : i + self.buffer_size]]
            vectors = np.concatenate(vectors, axis=0)
            total_data = self._update_id_mapping(db_ids)
            self.index.add(vectors)
            logger.info("data indexed %d", total_data)

        indexed_cnt = len(self.index_id_to_db_id)
        logger.info("Total data indexed %d", indexed_cnt)

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
        scores, indexes = self.index.search(query_vectors, top_docs)
        # convert to external ids
        db_ids = [[self.index_id_to_db_id[i] for i in query_top_idxs] for query_top_idxs in indexes]
        result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
        return result

    def get_index_name(self):
        return "flat_index"

class WikiSearcher():
    def __init__(self,model_file,index_path,wiki_passages):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_seq_len = 256
        self.question_encoder = HFBertEncoder.init_encoder(
            'bert-base-uncased',
            projection_dim=0,
            dropout=0.1,
            pretrained=True,
        )
        state_dict = torch.load(model_file)
        question_encoder_state_dict = OrderedDict({".".join(key.split(".")[1:]): value for key, value in state_dict["model_dict"].items() if
                     key.startswith("question_model")})
        self.question_encoder.load_state_dict(question_encoder_state_dict,strict=False)
        self.question_encoder.eval()

        vector_size = self.question_encoder.get_out_size()
        self.indexer = DenseFlatIndexer()
        # self.indexer = hydra.utils.instantiate({'_target_': 'cogktr.enhancers.searcher.dpr_searcher.DenseFlatIndexer'})

        # self.indexer = DenseFlatIndexer()
        self.indexer.init_index(vector_size)
        self.indexer.deserialize(index_path)

        wiki_infos = {}
        with open(wiki_passages) as ifile:
            reader = csv.reader(ifile,delimiter="\t")
            # row_count = sum(1 for row in ifile)
            pbar = tqdm(reader,total=21015324)
            for row in pbar:
                if row[0] == "id":
                    pbar.update(1)
                    continue
                sample_id = 'wiki:' + str(row[0])
                passage = row[1].strip('"')
                wiki_infos[sample_id] = passage
                pbar.update(1)

        self.wiki_infos = wiki_infos



    def search(self, query,n_doc=10):
        query = query.strip()
        encode_dict = self.tokenizer.encode_plus(query,
                                          add_special_tokens=True,
                                          max_length=self.max_seq_len,
                                          truncation=True,
                                          padding='max_length',)
        with torch.no_grad():
            _,out,_ = self.question_encoder(
                    torch.tensor([encode_dict['input_ids']]),
                    torch.tensor([encode_dict['token_type_ids']]),
                    torch.tensor([encode_dict['attention_mask']]),
                )

        # print("Searching...")
        results = self.indexer.search_knn(out.numpy(),top_docs=n_doc)
        search_result = []
        for idx,wiki_id in enumerate(results[0][0]):
            passage = self.wiki_infos[wiki_id]
            score = results[0][1][idx]
            search_result.append((wiki_id,passage,score))
        return search_result


if __name__ == '__main__':
    searcher = WikiSearcher(
        model_file='/data/hongbang/projects/DPR/downloads/checkpoint/retriever/single-adv-hn/nq/bert-base-encoder.cp',
        # model_file='/data/hongbang/projects/DPR/outputs/dpr_debug/encoder_state_dict.pt',
        index_path='/data/hongbang/projects/DPR/outputs/my_index/',
        wiki_passages='/data/hongbang/projects/DPR/downloads/data/wikipedia_split/psgs_w100.tsv')
    results =searcher.search("who is the mvp of golden warrior this year",n_doc=100)
    print("Searching Done!")
