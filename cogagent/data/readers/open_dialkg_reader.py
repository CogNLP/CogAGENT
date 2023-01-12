import os
from cogagent.data.readers.base_reader import BaseReader
from cogagent.data.datable import DataTable
from cogagent.utils.vocab_utils import Vocabulary
import json
from tqdm import tqdm
from collections import Counter
from cogagent.data.processors.open_dialkg_processors.graph import NER
from cogagent.data.processors.open_dialkg_processors.kge import KnowledgeGraphEmbedding,Triple
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from collections import OrderedDict

def _normalize(text: str) -> str:
    return " ".join(text.split())

class OpenDialKGReader(BaseReader):
    def __init__(self, raw_data_path,test_mode='seen',debug=False):
        super().__init__()
        if test_mode not in ['seen','unseen']:
            assert ValueError("Test mode must be seen or unseen but got {}!".format(test_mode))
        self.raw_data_path = raw_data_path
        self.train_file = 'train_nph_data.json'
        self.dev_file = 'dev_nph_data.json'
        self.test_file = 'test_nph_data.json'
        if debug:
            self.train_file = self.dev_file = self.test_file = "debug_nph_data.json"

        self.train_path = os.path.join(raw_data_path, self.train_file)
        self.dev_path = os.path.join(raw_data_path, self.dev_file)
        self.test_path = os.path.join(raw_data_path, self.test_file)

        # ner construction
        graph_file = os.path.join(self.raw_data_path,"opendialkg_triples.txt")
        if not os.path.exists(graph_file):
            graph_file = os.path.join(self.raw_data_path,"original_data/opendialkg_triples.txt")
            if not os.path.exists(graph_file):
                raise ValueError("Graph File {} could not be found in {}!".format("opendialkg_triples.txt",self.raw_data_path))
        self.ner = NER(self.raw_data_path,graph_file=graph_file)

        # kge construction
        self.kge = KnowledgeGraphEmbedding(os.path.join(self.raw_data_path,"graph_embedding"),self.train_path,self.dev_path,self.test_path)

        self.max_history = 3

    def _build_entity_ref(
        self, response: str, kb_triples: List[Triple], new_nodes: Set[str], new_rels: Set[str]
    ) -> Tuple[Optional[Dict[str, Tuple[str, str, int, str]]], int]:
        response_ents = [(ent, kg_ent) for ent, kg_ent, _ in self.ner.extract(response)]
        if not response_ents:
            return None, 0

        entity_refs = OrderedDict()
        unmatched_triples = set()
        seen_kg_ents = set()
        for ent, kg_ent in response_ents:
            if kg_ent in seen_kg_ents:
                continue
            seen_kg_ents.add(kg_ent)
            for t in range(len(kb_triples) - 1, -1, -1):
                triple = kb_triples[t]
                if ent.lower() == triple.subject.lower():
                    if kg_ent in self.ner.knowledge_graph[triple.object]:
                        entity_refs[ent] = (triple.object, triple.predicate, 0, kg_ent)
                        break
                    else:
                        unmatched_triples.add((triple.subject, triple.predicate, triple.object))
                elif ent.lower() == triple.object.lower():
                    if kg_ent in self.ner.knowledge_graph[triple.subject]:
                        entity_refs[ent] = (triple.subject, triple.predicate, 1, kg_ent)
                        break
                    else:
                        unmatched_triples.add((triple.subject, triple.predicate, triple.object))

        num_notfound_triples = len(unmatched_triples)

        if not entity_refs:
            return None, num_notfound_triples

        for ent, (pivot, *_) in entity_refs.items():
            for nbr, edges in self.ner.knowledge_graph[pivot].items():
                if not self.kge.contains_node(nbr):
                    new_nodes.add(nbr)
                for rel in edges.keys():
                    if not self.kge.contains_rel(rel):
                        new_rels.add(rel)

        return entity_refs, num_notfound_triples

    def _read(self,path=None):
        print("Reading data...")
        datable = DataTable()
        with open(path) as file:
            raw_lines = [json.loads(line) for line in file]

        new_nodes, new_rels = set(), set()
        num_notfound_triples = 0
        num_nce_instances = 0
        num_lm_instances = 0

        for dialogue in tqdm(raw_lines):
            dialogue["history"] = dialogue["history"][-self.max_history : ]
            dialogue["response"] = _normalize(dialogue["response"])
            if dialogue["knowledge_base"]:
                kb_triples = list(Triple.of(dialogue["knowledge_base"]["paths"]))
            else:
                continue

            if kb_triples:
                entity_refs, n_notfounds = self._build_entity_ref(dialogue["response"], kb_triples, new_nodes, new_rels)
                num_notfound_triples += n_notfounds
            else:
                entity_refs = None

            if entity_refs:
                num_nce_instances += 1

            num_lm_instances += 1

            dialogue["kb_triples"] = kb_triples
            dialogue["entities"] = entity_refs
            del dialogue["knowledge_base"]
            for key,value in dialogue.items():
                datable(key,value)
        self.kge.resize(new_nodes, new_rels)
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

        return {
            "kge":self.kge,
            "ner":self.ner,
        }

if __name__ == "__main__":
    from cogagent.utils.log_utils import init_logger
    logger = init_logger()
    reader = OpenDialKGReader(raw_data_path="/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/OpenDialKG/raw_data",debug=True)
    train_data,dev_data,test_data = reader.read_all()
    vocab = reader.read_vocab()

    # from cogagent import save_pickle
    # save_pickle(train_data,"/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/OpenDialKG/raw_data/train_data.pkl")