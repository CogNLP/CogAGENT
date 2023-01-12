import json
import os
from collections import defaultdict
from typing import Dict, Optional, Set, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Union
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from cogagent.data.processors.open_dialkg_processors.graph import refine_node

TokenizedText = List[str]
EncodedText = List[int]


@dataclass
class Triple:
    subject: Union[str, EncodedText]
    predicate: Union[str, EncodedText]
    object: Union[str, EncodedText]

    def encode(
        self, tokenizer: PreTrainedTokenizerBase, sep_token_id: Optional[int] = None, add_sep: bool = True
    ) -> EncodedText:
        if add_sep:
            sep_token_id = sep_token_id or tokenizer.sep_token_id

        return (
            self._may_encode("subject", tokenizer)
            + ([sep_token_id] if add_sep else [])
            + self._may_encode("predicate", tokenizer)
            + ([sep_token_id] if add_sep else [])
            + self._may_encode("object", tokenizer)
        )

    def _may_encode(self, attr: str, tokenizer: PreTrainedTokenizerBase) -> EncodedText:
        val = getattr(self, attr, None)
        if val is None:
            return []
        elif isinstance(val, str):
            return tokenizer.encode(val, add_special_tokens=False)
        else:
            return val

    @classmethod
    def of(cls, triples) -> Iterable["Triple"]:
        for triple in triples:
            if len(triple) < 3:
                continue

            s, p, o = refine_node(triple[0]), triple[1], refine_node(triple[2])
            if not s or not p or not o:
                continue

            if p.startswith("~"):
                p = p[1:]
                s, o = o, s

            yield Triple(s, p, o)

class KnowledgeGraphEmbedding:
    def __init__(self, kge_path: str, *dataset_paths, init_mean: float = 0.0, init_std: float = 1.0):
        self.kge_path = kge_path
        self.node2id, self.id2node = self._load_ids("entities.txt")
        self.rel2id, self.id2rel = self._load_ids("relations.txt")

        self.node_embds = self._load_and_stack("entity_embedding.npy")
        self.rel_embds = self._load_and_stack("relation_embedding.npy")

        if dataset_paths:
            new_ents = set()
            new_rels = set()
            for dataset_path in dataset_paths:
                if not dataset_path:
                    continue
                _ents, _rels = self._find_new_objs(dataset_path)
                new_ents.update(_ents)
                new_rels.update(_rels)

            if new_ents:
                self.node_embds = self._resize_embeddings(
                    new_ents, self.node2id, self.id2node, self.node_embds, init_mean, init_std
                )

            if new_rels:
                self.rel_embds = self._resize_embeddings(
                    new_rels, self.rel2id, self.id2rel, self.rel_embds, init_mean, init_std
                )

    def _load_and_stack(self, np_file: str):
        # breakpoint()
        embds = np.load(os.path.join(self.kge_path, np_file))
        return np.vstack((np.zeros(embds.shape[-1]), embds))

    def _load_ids(self, file_name: str) -> tuple:
        with open(os.path.join(self.kge_path, file_name)) as reader:
            ent2id = defaultdict(lambda: len(ent2id))
            ent2id[self.pad] = self.pad_id
            id2ent = {self.pad_id: self.pad}

            for line in reader:
                entity = line.strip()
                id = ent2id[entity]
                id2ent[id] = entity

        return dict(ent2id), id2ent

    @property
    def pad(self) -> str:
        return "<pad>"

    @property
    def pad_id(self):
        return 0

    def resize(
        self,
        new_ents: Optional[Set[str]] = None,
        new_rels: Optional[Set[str]] = None,
        dataset_path: str = None,
        init_mean: float = 0.0,
        init_std: float = 1.0,
    ) -> Tuple[int, int]:
        if dataset_path is None:
            assert new_ents is not None and new_rels is not None
        else:
            new_ents, new_rels = self._find_new_objs(dataset_path)

        if new_ents:
            self.node_embds = self._resize_embeddings(
                new_ents, self.node2id, self.id2node, self.node_embds, init_mean, init_std
            )

        if new_rels:
            self.rel_embds = self._resize_embeddings(
                new_rels, self.rel2id, self.id2rel, self.rel_embds, init_mean, init_std
            )

        return len(new_ents), len(new_rels)

    def _find_new_objs(self, dataset_path: str) -> tuple:
        new_ents = set()
        new_rels = set()
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                dialogue = json.loads(line.strip())
                if dialogue["knowledge_base"]:
                    for subj, pred, obj in dialogue["knowledge_base"]["paths"]:
                        if subj not in self.node2id:
                            new_ents.add(subj)
                        if obj not in self.node2id:
                            new_ents.add(obj)
                        if pred not in self.rel2id:
                            new_rels.add(pred)
        return new_ents, new_rels

    @classmethod
    def _resize_embeddings(
        cls,
        new_objs: Set[str],
        obj2id: Dict[str, int],
        id2obj: Dict[int, str],
        embds: np.ndarray,
        init_mean: float,
        init_std: float,
    ):
        new_embs = np.random.normal(init_mean, init_std, (len(new_objs), embds.shape[-1]))
        next_id = embds.shape[0]
        extended_embds = np.vstack((embds, new_embs))
        for obj in new_objs:
            obj2id[obj] = next_id
            id2obj[next_id] = obj
            next_id += 1

        return extended_embds

    def encode_node(self, node: str) -> int:
        return self.node2id[node]

    def decode_node(self, node_id: int) -> str:
        return self.id2node[node_id]

    def contains_node(self, node) -> bool:
        return node in self.node2id

    def encode_rel(self, relation: str) -> int:
        return self.rel2id[relation]

    def decode_rel(self, rel_id: int) -> str:
        return self.id2rel[rel_id]

    def contains_rel(self, relation: str) -> bool:
        return relation in self.rel2id

