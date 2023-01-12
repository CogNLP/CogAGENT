import logging
from typing import List, Tuple
from tqdm import tqdm

import networkx as nx
import spacy
import os
from pathlib import Path
import pickle

from cogagent.utils.log_utils import logger
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import spacy

class EntityTagger:

    skip_names: List[str] = ["he", "she", "you", "it", "or", "the woman", "remember", "yes", "one",
                             "something else", "popular", "book", "me", "two", "the way", "author",
                             "series", "may", "good", "they", "home", "her", "song", "saw", "love",
                             "hand", "sold", "nine", "for you", "good year", "star", "believe",
                             "anything else", "today", "bet", "born", "world", "film", "O", "watch",
                             "I am here", "family", "dad", "coach", "perfect", "mystery",
                             "information",
                             "up", "go", "co", "album", "film", "cult", "award", "blue",
                             "check it out",
                             "fiction", "princess", "two", "series", "second", "lot", "author",
                             "in The House"
                             "d", "room", "music", "white", "player", "novel", "artist", "play",
                             "about time",
                             "great day", "hope", "writing", "television", "play", "after", "news",
                             "always", "game",
                             "video", "page", "once", "a Major", "romance novel", "war films"]

    def __init__(self, graph: nx.Graph, output_path, verbose: bool=True):
        self.entities = {}
        self.nlp = spacy.load("en_core_web_sm", disable=("ner", "parser", "tagger", "lemmatizer"))

        output_path = Path(output_path)
        # breakpoint()
        if os.path.isfile(output_path / f"entity_types.pkl"):
            logger.info("Loading entities from the pickle file!")
            with open(output_path / f"entity_types.pkl", "rb") as f:
                self.entities = pickle.load(f)
        else:
            output_path.mkdir(parents=True, exist_ok=True)

            for entity in tqdm(graph.nodes, desc="Map KG nodes", disable=not verbose):
                ent = self._tokenize(entity).lower()
                if ent not in self.skip_names:
                    self.entities[self._tokenize(entity).lower()] = entity
            output_path = output_path / f"entity_types.pkl"
            with output_path.open("wb") as out_file:
                pickle.dump(self.entities, out_file)

    def _tokenize(self, text: str) -> str:
        return " ".join([tok.text for tok in self.nlp(text)])

    def extract(self, text: str) -> List[Tuple[str, str]]:
        tokens = self._tokenize(text).split()

        entities = []
        i = 0
        while i < len(tokens):
            longest_match = None
            tail_index = -1
            for j in range(i, len(tokens)):
                cand = " ".join(tokens[i:j + 1])
                node = self.entities.get(cand.lower(), None)
                if node:
                    longest_match = (cand, node)
                    tail_index = j + 1

            if longest_match:
                entities.append(longest_match)
                i = tail_index
            else:
                i += 1

        return entities


def refine_node(node: str) -> str:
    if node == "Megamind":
        node = "MegaMind"

    return node


def load_kg(freebase_file: str, verbose: bool=True) -> nx.Graph:
    G = nx.MultiGraph()
    incomplete_triples = 0
    total_triples = 0

    with open(freebase_file, "r") as f:
        for line in f.readlines():
            total_triples += 1
            if len(line.strip().split("\t")) < 3:
                incomplete_triples += 1
                continue
            head, edge, tail = line.strip().split("\t")
            head = refine_node(head)
            tail = refine_node(tail)
            if edge.startswith("~"):
                edge = edge[1:]
                src, dest = tail, head
            else:
                src, dest = head, tail
            
            if not G.has_edge(src, dest, key=edge):
                G.add_edge(src, dest, key=edge)

    if verbose:
        logger.info("Number of incomplete triples {} out of {} total triples".format(incomplete_triples, total_triples))
        logger.info("Number of nodes: {} | Number of edges: {}".format(G.number_of_nodes(), G.number_of_edges()))

    return G


class NER:
    def __init__(self, output_path, method: str = "kg", graph_file: str = None, knowledge_graph: nx.Graph = None):
        logger.info(f"loading NER model using {method}...")
        if knowledge_graph is None:
            assert graph_file is not None
            self.knowledge_graph = load_kg(graph_file)
        else:
            self.knowledge_graph = knowledge_graph

        if method == "spacy":
            self._nlp = spacy.load("en_core_web_sm")
        elif method == "kg":
            self._graph_tagger = EntityTagger(self.knowledge_graph, output_path)
        else:
            raise ValueError(f"Unknown NER method: `{method}`")
        self.method = method

    def extract(self, text: str) -> List[Tuple[str, str, Optional[str]]]:
        if self.method == "spacy":
            return self._spacy_extract(text)
        else:
            return self._graph_extract(text)

    def _spacy_extract(self, text: str) -> List[Tuple[str, str, str]]:
        tokens = self._nlp(text)

        ents = []

        ent, ent_type = [], []
        start_offset, end_offset = 0, 0
        for tok in tokens:
            if tok.ent_iob_ == "B":
                if ent:
                    if text[start_offset:end_offset] in self.knowledge_graph:
                        ents.append((" ".join(ent), ent_type[0]))
                ent.append(tok.text)
                ent_type.append(tok.ent_type_)
                start_offset = tok.idx
                end_offset = start_offset + len(tok.text)
            elif tok.ent_iob_ == "I":
                ent.append(tok.text)
                ent_type.append(tok.ent_type_)
                end_offset = tok.idx + len(tok.text)

        if ent:
            if text[start_offset:end_offset] in self.knowledge_graph:
                ents.append((" ".join(ent), text[start_offset:end_offset], ent_type[0]))

        return ents

    def _graph_extract(self, text: str) -> List[Tuple[str, str, Optional[str]]]:
        return [(ent, kg_key, None) for ent, kg_key in self._graph_tagger.extract(text)]


