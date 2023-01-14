from cogagent.data.datable import DataTable
from cogagent.data.datableset import DataTableSet
from tqdm import tqdm
import transformers
from cogagent.data.processors.base_processor import BaseProcessor
transformers.logging.set_verbosity_error()  # set transformers logging level
from transformers import AutoTokenizer
import spacy
from cogagent.utils.constant.opendialkg_constants import SPECIAL_TOKENS,ATTR_TO_SPECIAL_TOKEN,SpecialTokens
from cogagent.data.processors.open_dialkg_processors.kge import Triple,EncodedText,TokenizedText
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from collections import OrderedDict
import re
import numpy as np
import torch
from cogagent.utils.log_utils import logger

def _normalize(text: str) -> str:
    return " ".join(text.split())

class OpenDialKGForNPHProcessor(BaseProcessor):
    def __init__(self, vocab,plm,mlm,debug):
        super().__init__(debug)
        self.ner = vocab["ner"]
        self.kge = vocab["kge"]

        self.tokenizer = AutoTokenizer.from_pretrained(plm)
        self.mlm_tokenizer = AutoTokenizer.from_pretrained(mlm)

        num_added_tokens = self.tokenizer.add_special_tokens(
            ATTR_TO_SPECIAL_TOKEN
        )
        vocab["new_num_tokens"] = len(self.tokenizer)
        self.special_tokens = SpecialTokens(*self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS))
        self.mlm_special_tokens = SpecialTokens(*self.mlm_tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS))

        self.nlp = spacy.load("en_core_web_sm", disable=("ner", "parser", "tagger", "lemmatizer"))

        self.include_render = False
        self.exclude_kb = False
        self.max_adjacents = 100

        self.padding = 'longest'
        self.pad_to_multiple_of = 8
        self.kg_pad = 0
        self.mask_pad = 0
        self.label_pad = -100

        self.fields: Iterable[str] = (
            "input_ids",
            "token_type_ids",
            "attention_mask",
            "triple_ids",
            "triple_type_ids",
            "lm_labels",
            "subject_ids",
            "predicate_ids",
            "object_ids",
        )

    @property
    def special_pad_tokens(self) -> Dict[str, int]:
        return dict(
            attention_mask=self.mask_pad,
            lm_labels=self.label_pad,
            subject_ids=self.kg_pad,
            predicte_ids=self.kg_pad,
            object_ids=self.kg_pad,
        )


    def _word_tokenize(self, text: str, *, as_string: bool = True) -> Union[str, List[str]]:

        # To resolve issues like 'He also directed Batman R.I.P.. Have you seen that one?'
        text = re.sub(r"(\w+\.)\.\s", r"\1 . ", text)
        text = re.sub(r"(\.\w\.)\.\s", r"\1 . ", text)

        # To resolve issues like 'I like Neil Brown Jr..' and 'I think he plays for Real Madrid C.F..'
        if re.match(r".*\w+\.\.$", text) or re.match(r".*\.\w\.\.$", text):
            text = text[:-1] + " ."

        tokens = [tok.text for tok in self.nlp(text)]
        if as_string:
            return " ".join(tokens)
        else:
            return tokens

    def _build_from_segments(
        self,
        kb_triples: List[Triple],
        render: Optional[List[int]],
        history: List[List[int]],
        speaker: int,
        response: List[int],
        with_eos: bool = True,
    ) -> Dict[str, List[int]]:
        """ Builds a sequence of input from 3 segments: history, kb triples and response. """

        tokens, token_types = [self.special_tokens.bos], [self.special_tokens.bos]
        # KB
        token_ids_triples = []
        token_types_triples = []
        if kb_triples is not None:
            for triple in kb_triples:
                if self.kge is None:
                    token_ids_triples.extend(
                        [self.special_tokens.subject]
                        + triple.subject
                        + [self.special_tokens.predicate]
                        + triple.predicate
                        + [self.special_tokens.object]
                        + triple.object
                    )
                    token_types_triples.extend(
                        [self.special_tokens.subject] * (len(triple.subject) + 1)
                        + [self.special_tokens.predicate] * (len(triple.predicate) + 1)
                        + [self.special_tokens.object] * (len(triple.object) + 1)
                    )
                else:
                    token_ids_triples.extend([self.tokenizer.pad_token_id] * 3)
                    token_types_triples.extend(
                        [self.special_tokens.subject, self.special_tokens.predicate, self.special_tokens.object]
                    )

        if render:
            token_ids_render = render
            token_types_render = [self.special_tokens.triple] * len(render)
        else:
            token_ids_render = []
            token_types_render = []

        # History
        token_id_history = []
        token_type_history = []
        sequence = history + ([response] if with_eos else [])

        if len(history) % 2 == 1:
            current_speaker = self._other_speaker(speaker)
        else:
            current_speaker = speaker

        for i, utterance in enumerate(sequence):
            token_id_history.extend([current_speaker] + utterance)
            token_type_history.extend([current_speaker] * (len(utterance) + 1))
            current_speaker = self._other_speaker(current_speaker)

        tokens.extend(token_ids_triples + token_ids_render + token_id_history)
        token_types.extend(token_types_triples + token_types_render + token_type_history)

        # For training
        if with_eos:
            tokens.append(self.special_tokens.eos)
            token_types.append(self._other_speaker(current_speaker))

            labels = {
                "lm_labels": (
                    [-100]
                    + [-100] * (len(token_ids_triples) + len(token_id_history) - len(response))
                    + response
                    + [self.special_tokens.eos]
                )
            }

        # For testing
        else:
            labels = {}
            tokens.append(current_speaker)
            token_types.append(current_speaker)

        return dict(
            input_ids=tokens,
            token_type_ids=token_types,
            attention_mask=[1] * len(tokens),
            triple_ids=token_ids_triples,
            triple_type_ids=token_types_triples,
            **labels,
            # current_speaker=current_speaker,
        )

    def _other_speaker(self, speaker: int):
        if speaker == self.special_tokens.speaker1:
            return self.special_tokens.speaker2
        else:
            return self.special_tokens.speaker1

    def _build_dialogue_lm(
        self, history: List[str], render: Optional[str], response: str, kb_triples: List[Triple], speaker: int
    ) -> Dict[str, List[Any]]:
        encoded_history = [self.tokenizer.encode(h, add_special_tokens=False) for h in history]
        encoded_response = self.tokenizer.encode(response, add_special_tokens=False)

        triples = []
        if not self.exclude_kb:
            for triple in kb_triples:
                encoded_triple = (
                    self.tokenizer.encode(triple.subject, add_special_tokens=False),
                    self.tokenizer.encode(triple.predicate, add_special_tokens=False),
                    self.tokenizer.encode(triple.object, add_special_tokens=False),
                )

                triples.append(Triple(*encoded_triple))

        if self.include_render and render is not None:
            encoded_render = self.tokenizer.encode(render, add_special_tokens=False)
        else:
            encoded_render = None

        return self._build_from_segments(
            triples,
            encoded_render,
            encoded_history,
            speaker,
            encoded_response,
            with_eos=True,
        )


    def _build_from_graph(
        self, entities: Dict[str, Tuple[str, str, int, str]]
    ) -> Tuple[List[List[int]], List[List[int]], List[int], List[int], List[int], List[int]]:
        neighbors = []
        rels = []
        pivots = []
        pivot_fields = []
        labels = []
        label_indices = []

        for ent, (pivot_ent, pivot_rel, pivot_field, kg_ent) in entities.items():
            pivot_id = self.kge.encode_node(pivot_ent)
            pivots.append(pivot_id)
            pivot_fields.append(pivot_field)

            ent_neighbors = {}
            neighbors_real_int = {}
            ent_id = self.kge.encode_node(kg_ent)
            for object, edges in self.ner.knowledge_graph[pivot_ent].items():
                object_id = self.kge.encode_node(object)

                for rel in edges.keys():
                    if object == kg_ent and rel != pivot_rel:
                        continue

                    # if rel == pivot_rel:
                    # breakpoint()
                    # continue
                    neighbors_real_int[object] = rel
                    rel_id = self.kge.encode_rel(rel)
                    ent_neighbors[object_id] = rel_id

                if object == kg_ent and ent_id not in ent_neighbors:
                    rel = next(iter(edges.keys()))
                    logger.warning(
                        f"`{pivot_rel}` not found and replaced with `{rel}` for `{object}`, neighbor of `{pivot_ent}`"
                    )

                    rel_id = self.kge.encode_rel(rel)
                    ent_neighbors[object_id] = rel_id
                    neighbors_real_int[object] = rel

            if ent_id not in ent_neighbors:
                raise ValueError(
                    f"`{kg_ent}` ({ent_id}) appeared as `{ent}` not found as a neighbor of `{pivot_ent}` "
                    f"with relation with relation `{pivot_rel}`: {ent_neighbors}"
                )

            nbr_list = [nbr for nbr, _ in ent_neighbors.items()]
            if self.max_adjacents > 0 and len(nbr_list) > self.max_adjacents:
                if self.max_adjacents > 1:
                    nbr_list.remove(ent_id)
                    nbr_list = np.random.choice(nbr_list, size=self.max_adjacents - 1, replace=False).tolist()
                    nbr_list.insert(np.random.randint(len(nbr_list)), ent_id)
                else:
                    nbr_list = [ent_id]

            neighbors.append(nbr_list)
            label_indices.append(nbr_list.index(ent_id))
            labels.append(ent_id)

            rels.append([ent_neighbors[nbr] for nbr in nbr_list])
        return neighbors, rels, pivots, pivot_fields, labels, label_indices


    def mask_entity(self, tokens: TokenizedText, entity: TokenizedText) -> Tuple[TokenizedText, List[int]]:
        assert len(entity) > 0

        masked_tokens = list(tokens)
        entity_mask = [0] * len(tokens)
        i = 0
        while i < (len(tokens) - len(entity) + 1):
            if entity == tokens[i : i + len(entity)]:
                for j in range(i, i + len(entity)):
                    masked_tokens[j] = self.mlm_tokenizer.mask_token
                    entity_mask[j] = 1
                i += len(entity)
            else:
                i += 1
        return masked_tokens, entity_mask


    def mask_and_collapse_entity(
        self, tokens: TokenizedText, entity: TokenizedText, pad_indices: List[int]
    ) -> Tuple[TokenizedText, List[int]]:
        assert len(entity) > 0

        masked_sequence = []
        i = 0
        pad_index = -1
        while i < len(tokens):
            if entity == tokens[i : i + len(entity)]:
                masked_sequence.append(self.tokenizer.pad_token)
                if pad_index < 0:
                    pad_index = i
                    pad_indices.append(pad_index)
                pad_indices = [pi if pi <= i or pi < 0 else (pi - len(entity) + 1) for pi in pad_indices]
                i += len(entity)
            else:
                masked_sequence.append(tokens[i])
                i += 1

        if pad_index < 0:
            pad_indices.append(pad_index)

        return masked_sequence, pad_indices

    def _build_mlm_from_segments(
        self,
        render: Optional[str],
        history: List[str],
        triples: List[Triple],
        response: str,
        entities: Iterable[str],
        ordered_mask: List[int],
    ) -> dict:
        mlm_tokens = [self.mlm_tokenizer.bos_token_id]
        for utterance in history:
            mlm_tokens.extend(self.mlm_tokenizer.encode(utterance, add_special_tokens=False))

        mlm_tokens.append(self.mlm_tokenizer.sep_token_id)

        if not self.exclude_kb:
            for triple in triples:
                mlm_tokens.extend(triple.encode(self.mlm_tokenizer))
                mlm_tokens.append(self.mlm_tokenizer.sep_token_id)

        if self.include_render and render is not None:
            mlm_tokens.extend(self.mlm_tokenizer.encode(render, add_special_tokens=False))
            mlm_tokens.append(self.mlm_tokenizer.sep_token_id)

        mlm_response = self.mlm_tokenizer.tokenize(response)
        response_entity_mask = [0] * len(mlm_response)

        for i, ent in enumerate(entities):
            mlm_tokenized_ent = self.mlm_tokenizer.tokenize(" " + ent)
            if not mlm_tokenized_ent:
                raise ValueError(f"Entity `{ent}` tokenized to empty!")

            mlm_response, mlm_curr_entity_mask = self.mask_entity(mlm_response, mlm_tokenized_ent)
            response_entity_mask = [
                m or ((ordered_mask[i] + 1) if n else 0) for m, n in zip(response_entity_mask, mlm_curr_entity_mask)
            ]

        if all(m == 0 for m in response_entity_mask):
            raise ValueError("Entity mask must not contain all zeros")

        if len(response_entity_mask) != len(mlm_response):
            raise ValueError("Entity mask and response do not have the same length")

        entity_mask = [0] * len(mlm_tokens) + response_entity_mask
        mlm_tokens.extend(self.mlm_tokenizer.convert_tokens_to_ids(mlm_response))

        return dict(
            input_ids=mlm_tokens,
            attention_mask=[1] * len(mlm_tokens),
            response_entity_mask=response_entity_mask,
            entity_mask=entity_mask,
        )


    def _build_lm_from_segments(
        self, render: Optional[str], history: List[str], triples: List[Triple], response: str, entities: Iterable[str]
    ) -> dict:
        lm_tokens = [self.tokenizer.bos_token_id]
        for utterance in history:
            lm_tokens.extend(self.tokenizer.encode(utterance, add_special_tokens=False))

        lm_tokens.append(self.tokenizer.sep_token_id)

        if not self.exclude_kb:
            for triple in triples:
                lm_tokens.extend(triple.encode(self.tokenizer))
                lm_tokens.append(self.tokenizer.sep_token_id)

        if self.include_render and render is not None:
            lm_tokens.extend(self.tokenizer.encode(render, add_special_tokens=False))
            lm_tokens.append(self.tokenizer.sep_token_id)

        lm_response = self.tokenizer.tokenize(response)

        pad_indices = []
        for i, ent in enumerate(entities):
            lm_tokenized_ent = self.tokenizer.tokenize(" " + ent)
            if not lm_tokenized_ent:
                raise ValueError(f"Entity `{ent}` tokenized to empty!")

            lm_response, pad_indices = self.mask_and_collapse_entity(lm_response, lm_tokenized_ent, pad_indices)

        lm_attention_masks = []
        for pad_index in pad_indices:
            if pad_index >= 0:
                attn_mask = [1] * (pad_index + 1) + [0] * (len(lm_response) - pad_index - 1)
            else:
                attn_mask = [0] * len(lm_response)

            lm_attention_masks.append([1] * len(lm_tokens) + attn_mask)

        lm_tokens.extend(self.tokenizer.convert_tokens_to_ids(lm_response))

        return dict(
            input_ids=lm_tokens,
            attention_mask = lm_attention_masks,
        )

    def _pad3D(self, examples: List[dict], attr: str, E: int, C: int, pad: int = None):
        padded = np.full((len(examples), E, C), pad or self.kg_pad, dtype=int)

        for i, ex in enumerate(examples):
            for j, cands in enumerate(ex[attr]):
                padded[i, j, : len(cands)] = cands

        return torch.from_numpy(padded)

    def _pad2D(self, examples: List[dict], attr: str, E: int, pad: int):
        padded = np.full((len(examples), E), pad, dtype=int)

        for i, ex in enumerate(examples):
            labels = ex[attr]
            padded[i, : len(labels)] = labels

        return torch.from_numpy(padded)

    def _get_pad_token(self, field: str) -> int:
        pad = self.special_pad_tokens.get(field, None)
        return self.tokenizer.pad_token_id if pad is None else pad

    def _collate(self, batch):
        mask_refine_examples = [sample["mask_refine_example"] for sample in batch if sample["mask_refine_example"]]
        lm_examples = [sample["lm_inputs"] for sample in batch if sample["lm_inputs"]]

        nce_batch,lm_batch = None,None

        if mask_refine_examples:
            lm_input_ids = self.tokenizer.pad(
                [{"input_ids": sample["lm_input_ids"]} for sample in mask_refine_examples],
                padding = self.padding,
                pad_to_multiple_of = self.pad_to_multiple_of,
                return_tensors ='pt',
            )["input_ids"]

            max_ents = max([len(ex["lm_attention_mask"]) for ex in mask_refine_examples])
            max_seq_l = max([len(m) for ex in mask_refine_examples for m in ex["lm_attention_mask"]])
            if self.pad_to_multiple_of:
                max_seq_l = int(self.pad_to_multiple_of * np.ceil(max_seq_l / self.pad_to_multiple_of))

            padded_attention_mask = np.zeros((len(mask_refine_examples), max_ents, max_seq_l), dtype=int)
            for i, ex in enumerate(mask_refine_examples):
                for j, m in enumerate(ex["lm_attention_mask"]):
                    padded_attention_mask[i, j, : len(m)] = m

            # mlm padding
            max_mlm_l = max([len(ex["mlm_entity_mask"]) for ex in mask_refine_examples])
            if self.pad_to_multiple_of:
                max_mlm_l = int(self.pad_to_multiple_of * np.ceil(max_mlm_l / self.pad_to_multiple_of))

            padded_mlm_entity_mask = np.zeros((len(mask_refine_examples), max_mlm_l), dtype=int)
            for i, ex in enumerate(mask_refine_examples):
                padded_mlm_entity_mask[i, : len(ex["mlm_entity_mask"])] = ex["mlm_entity_mask"]

            # label padding
            contains_labels = all(ex["labels"] is not None for ex in mask_refine_examples)
            if contains_labels:
                max_label_l = max([len(ex["labels"]) for ex in mask_refine_examples])
                assert max_label_l == max_ents
            else:
                max_label_l = max([len(ex["candidate_ids"]) for ex in mask_refine_examples])

            max_cands = max([len(cands) for ex in mask_refine_examples for cands in ex["candidate_ids"]])
            padded_candidate_ids = self._pad3D(mask_refine_examples, "candidate_ids", max_label_l, max_cands)
            padded_candidate_rels = self._pad3D(mask_refine_examples, "candidate_rels", max_label_l, max_cands)
            padded_pivots = self._pad2D(mask_refine_examples, "pivot_ids", max_label_l, self.kg_pad)
            padded_pivot_fields = None

            mlm_batch = self.mlm_tokenizer.pad(
                [{"input_ids":ex["mlm_input_ids"],"attention_mask":ex["mlm_attention_mask"]} for ex in mask_refine_examples],
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors="pt",
            )
            padded_labels = (
                self._pad2D(mask_refine_examples, "labels", max_label_l, self.label_pad) if contains_labels else None
            )
            padded_label_indices = self._pad2D(mask_refine_examples, "label_indices", max_label_l, self.label_pad) if contains_labels else None

            nce_batch = dict(
                mlm_inputs = {**mlm_batch},
                mlm_entity_mask = torch.from_numpy(padded_mlm_entity_mask) if padded_mlm_entity_mask is not None else None,
                lm_input_ids = lm_input_ids,
                lm_attention_masks = torch.from_numpy(padded_attention_mask),
                candidate_ids = padded_candidate_ids,
                candidate_rels = padded_candidate_rels,
                pivot_ids = padded_pivots,
                pivot_fields = padded_pivot_fields,
                labels = padded_labels,
                label_indices = padded_label_indices,
            )

        if lm_examples:
            batch = lm_examples
            bsz = len(batch)
            padded_batch = {}
            # Go over each data point in the batch and pad each example.
            for name in self.fields:
                if any(name not in x for x in batch):
                    continue

                max_l = max(len(x[name]) for x in batch)
                if self.pad_to_multiple_of:
                    max_l = int(self.pad_to_multiple_of * np.ceil(max_l / self.pad_to_multiple_of))

                pad_token = self._get_pad_token(name)

                # Fill the batches with padding tokens
                padded_field = np.full((bsz, max_l), pad_token, dtype=np.int64)

                # batch is a list of dictionaries
                for bidx, x in enumerate(batch):
                    padded_field[bidx, : len(x[name])] = x[name]

                padded_batch[name] = torch.from_numpy(padded_field)

            lm_batch = padded_batch


        return dict(
            nce_batch = nce_batch,
            lm_batch = lm_batch,
        )


    def _process(self, data,is_training=True):
        datable = DataTable()
        data = self.debug_process(data)
        print("Processing data...")

        for dialog in tqdm(data):
            history,response,speaker,dialogue_id,kb_triples,entities = dialog

            if speaker.lower() in ['user','wizard']:
                speaker = self.special_tokens.speaker2
            else:
                speaker = self.special_tokens.speaker1

            render = None

            # start encoding
            response = self._word_tokenize(" " + response)
            history = [self._word_tokenize(u) for u in history]

            # # build lm input
            lm_inputs = self._build_dialogue_lm(history, render, response, kb_triples, speaker)

            # # build mask refine input
            mask_refine_example = None
            if entities:
                neighbors, rels, pivots, pivot_fields, labels, label_indices = self._build_from_graph(entities)
                ordered_entities = sorted(
                    entities.keys(), key=lambda x: (len(x.split()), len(x)), reverse=True
                )
                entities = list(entities.keys())
                ordered_mask = [entities.index(ent) for ent in ordered_entities]

                if self.mlm_tokenizer is not None:
                    mlm_example = self._build_mlm_from_segments(
                        render, history, kb_triples, response, ordered_entities, ordered_mask
                    )
                else:
                    mlm_example = None

                lm_example = self._build_lm_from_segments(
                    render, history, kb_triples, response, ordered_entities
                )
                mask_refine_example = dict(
                    mlm_input_ids = mlm_example["input_ids"],
                    mlm_attention_mask = mlm_example["attention_mask"],
                    mlm_entity_mask = mlm_example["entity_mask"],
                    lm_input_ids = lm_example["input_ids"],
                    lm_attention_mask = lm_example["attention_mask"],
                    candidate_ids = neighbors,
                    candidate_rels = rels,
                    pivot_ids = pivots,
                    pivot_fields = pivot_fields,
                    labels = labels,
                    label_indices = label_indices,
                )

            datable("lm_inputs",lm_inputs)
            datable("mask_refine_example",mask_refine_example)
        return DataTableSet(datable)

    def process_train(self, data):
        return self._process(data)

    def process_dev(self, data):
        return self._process(data,is_training=False)

    def process_test(self, data):
        return self._process(data,is_training=False)


if __name__ == "__main__":
    from cogagent.data.readers.open_dialkg_reader import OpenDialKGReader
    from cogagent.utils.log_utils import init_logger
    logger = init_logger()
    reader = OpenDialKGReader(raw_data_path="/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/OpenDialKG/raw_data",debug=True)
    # train_data,dev_data,test_data = reader.read_all()
    train_data = reader._read_train()
    vocab = reader.read_vocab()

    # change to roberta-large when running
    processor = OpenDialKGForNPHProcessor(vocab=vocab,plm='gpt2',mlm='roberta-large',debug=True)
    train_dataset = processor.process_train(train_data)

    # test collate function
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset=train_dataset, batch_size=8, collate_fn=processor._collate)
    sample = next(iter(dataloader))
    from cogagent.utils.train_utils import move_dict_value_to_device
    move_dict_value_to_device(sample,device=torch.device("cuda:5"))









