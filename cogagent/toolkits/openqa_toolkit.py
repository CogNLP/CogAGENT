import os.path

from cogagent.toolkits.base_toolkit import BaseToolkit
from cogagent.modules.search_modules import WikiSearcher
from cogagent.utils.log_utils import logger
from cogagent.data.datable import DataTable
from cogagent.data.datableset import DataTableSet
from cogagent.utils.train_utils import move_dict_value_to_device
from cogagent import save_pickle, load_pickle,load_model
import torch
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import collections
from argparse import Namespace
from torch.utils.data import DataLoader
from cogagent.models.base_reading_comprehension_model import BaseReadingComprehensionModel
from collections import  defaultdict
import string

import re
import torch
import collections
from torch._six import string_classes
np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


class OpenqaAgent(BaseToolkit):
    def __init__(self, bert_model,model_path, device,retriever_model_file,retriever_index_path,wiki_passages,debug=False):
        super(OpenqaAgent, self).__init__(bert_model,model_path,None,device)
        self.debug = debug

        logger.info("Constructing wikipedia searcher...")
        self.searcher = WikiSearcher(
            model_file=retriever_model_file,
            index_path=retriever_index_path,
            wiki_passages=wiki_passages
        )
        # self.searcher = WikiSearcher(
        #     model_file='/data/hongbang/projects/DPR/downloads/checkpoint/retriever/single-adv-hn/nq/bert-base-encoder.cp',
        #     index_path='/data/hongbang/projects/DPR/outputs/my_index/',
        #     wiki_passages='/data/hongbang/projects/DPR/downloads/data/wikipedia_split/psgs_w100.tsv'
        # )

        self.max_token_len = 384
        self.batch_size = 4
        self.n_best_size = 20
        self.max_answer_length = 30

        self.model = BaseReadingComprehensionModel(plm=bert_model,vocab=None)
        self.model = load_model(self.model,self.model_path)
        self.model.to(self.device)

    def run(self):
        while True:
            question = self.get_user_input()
            if question == 'q':
                break
            wiki_psgs = self.search_wiki(question)
            dataloader = self.construct_dataloader(question, wiki_psgs)
            _,pred_text = self.predict(dataloader)
            print("Agent>>",pred_text)
        print("Agent>>Thanks for using open domain QA system!")

    def predict(self,dataloader):
        qas_id2info = defaultdict(list)
        qas_id2example = {}

        with torch.no_grad():
            for batch in dataloader:
                move_dict_value_to_device(batch, device=self.device)
                start_logits,end_logits,batch = self.model.predict(batch)

                for i, example in enumerate(batch["example"]):
                    feature1 = {
                        key: value[i].cpu().tolist() for key, value in batch.items() if
                        key != "example" and key != "additional_info" and value is not None
                    }
                    feature2 = vars(batch["additional_info"][i])
                    qas_id2info[example.qas_id].append({
                        "start_logits": start_logits[i],
                        "end_logits": end_logits[i],
                        "feature": Namespace(**feature1, **feature2),
                    })
                    qas_id2example[example.qas_id] = example

        question_id2prediction = defaultdict(list)

        Prediction =collections.namedtuple(
            "prediction",["text","start_logit","end_logit","score"]
        )
        results = []
        for index,(qas_id,example) in enumerate(qas_id2example.items()):
            # info = self.qas_id2info[qas_id]
            infos = qas_id2info[qas_id]
            score_null = 1000000
            null_start_logit = 0
            null_end_logit = 0
            seen_predictions = {}
            n_best  =[]
            for (info_index, info) in enumerate(infos):
                start_logits = info["start_logits"]
                end_logits = info["end_logits"]
                feature = info["feature"]
                start_indexes = _get_best_indexes(start_logits, self.n_best_size)
                end_indexes = _get_best_indexes(end_logits, self.n_best_size)
                feature_null_score = start_logits[0] + end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    null_start_logit = start_logits[0]
                    null_end_logit = end_logits[0]
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if start_index >= len(feature.tokens):
                            continue
                        if end_index >= len(feature.tokens):
                            continue
                        if start_index not in feature.token_to_orig_map:
                            continue
                        if end_index not in feature.token_to_orig_map:
                            continue
                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > self.max_answer_length:
                            continue
                        # after obtain one valid span:
                        tok_tokens = feature.tokens[start_index:(end_index + 1)]
                        tok_text = " ".join(tok_tokens)
                        tok_text = tok_text.replace(" ##", "")
                        tok_text = tok_text.replace("##", "")
                        tok_text = tok_text.strip()
                        tok_text = " ".join(tok_text.split())
                        final_text = tok_text
                        if final_text in seen_predictions:
                            continue
                        seen_predictions[final_text] = True
                        n_best.append(
                            Prediction(
                                text=final_text,
                                start_logit=start_logits[start_index],
                                end_logit=end_logits[end_index],
                                score=start_logits[start_index]+end_logits[end_index],
                            )
                        )

            n_best.append(Prediction(
                text = "",
                start_logit=null_start_logit,
                end_logit=null_end_logit,
                score=null_start_logit+null_end_logit
            ))
            n_best = sorted(
                n_best,
                key=lambda x:(x.start_logit+x.end_logit),
                reverse=True, # Descending Order
            )
            n_best = n_best if len(n_best) < self.n_best_size else n_best[:self.n_best_size]
            question_id2prediction[qas_id.split('-')[0]].append(n_best[0])

        question_id2em_score = {}
        question_id2f1_score = {}
        assert len(question_id2prediction) == 1
        for question_id,value in question_id2prediction.items():
            sorted_values=sorted(
                value[:10] if len(value) > 10 else value,
                key=lambda x:(x.start_logit+x.end_logit),
                reverse=True, # Descending Order
            )
            pred_text = sorted_values[0].text
            example = qas_id2example[str(question_id)+'-'+'0']
            gold_answers = [a['text'] for a in example.answers if normalize_answer(a['text'])]
            if not gold_answers:
                # For unanswerable questions, only correct answer is empty string
                gold_answers = ['']
            results.append((example.question_text,pred_text))
        return results[0]

    def construct_dataloader(self, question, wiki_psgs):
        datable = DataTable()
        question_id = 0
        passages = wiki_psgs
        answers = []
        for psg_id, psg in enumerate(passages):
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            context_text = psg[1]

            # Split on whitespace so that different tokens may be attributed to their original position.
            for c in context_text:
                if _is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)
            example = Namespace(**{"question_text": question,
                                   "doc_tokens": doc_tokens,
                                   "qas_id": str(question_id) + '-' + str(psg_id),
                                   "answers": [{'text': answer} for answer in answers],
                                   "is_impossible": False,
                                   "relavance_score": psg[2],
                                   })
            results = prepare_mrc_input(
                example,
                tokenizer=self.tokenizer,
                max_seq_length=self.max_token_len,
                doc_stride=128,
                max_query_length=64,
                is_training=False
            )
            for result in results:
                for key, value in result.items():
                    datable(key, value)
                datable("example", example)
        datable.add_not2torch("example")
        datable.add_not2torch("additional_info")

        dataset = DataTableSet(datable)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            drop_last=False,
            collate_fn=self._collate
        )
        return dataloader

    def get_user_input(self, ):
        if self.debug:
            input_msg = "Which part of earth is covered with water?"
            print("User>>{}".format(input_msg))
        else:
            input_msg = input("User>>")
        return input_msg

    def search_wiki(self, question, n_doc=20):
        results = self.searcher.search(question, n_doc=n_doc)
        return results

    def _collate(self, batch):
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage).resize_(len(batch), *list(elem.size()))
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(default_collate_err_msg_format.format(elem.dtype))

                return self._collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, collections.abc.Mapping):
            try:
                return elem_type({key: self._collate([d[key] for d in batch]) for key in elem})
            except TypeError:
                # The mapping type may not support `__init__(iterable)`.
                return {key: self._collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self._collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError('each element in list of batch should be of equal size')
            transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

            if isinstance(elem, tuple):
                return [self._collate(samples) for samples in transposed]  # Backwards compatibility.
            else:
                try:
                    return elem_type([self._collate(samples) for samples in transposed])
                except TypeError:
                    # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                    return [self._collate(samples) for samples in transposed]

        elif isinstance(elem,Namespace):
            return batch

def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False



def prepare_mrc_input(example, tokenizer, max_seq_length, doc_stride, max_query_length, is_training):
    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            example.answer_text)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    _DocSpan = collections.namedtuple(
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    results = []
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                   split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        start_position = None
        end_position = None
        if is_training and not example.is_impossible:
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (tok_start_position >= doc_start and
                    tok_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
            else:
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
        if is_training and example.is_impossible:
            start_position = 0
            end_position = 0
        additional_info = Namespace(**{
            "token_to_orig_map": token_to_orig_map,
            "token_is_max_context": token_is_max_context,
            "tokens": tokens,
        })
        results.append({
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": segment_ids,
            "start_position": start_position,
            "end_position": end_position,
            "additional_info": additional_info,
        })

    return results


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start: (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def normalize_answer(s):

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

if __name__ == '__main__':
    import yaml
    import os

    with open("/data/hongbang/CogAGENT/demo/pages/config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    for key,value in config["openqa"].items():
        path = os.path.join(config["rootpath"],value)
        if os.path.exists(path):
            config["openqa"][key] = path

    agent = OpenqaAgent(
        bert_model="bert-base-uncased",
        device=torch.device("cuda:9"),
        **config["openqa"],
    )
    # agent = OpenqaAgent(
    #     bert_model="bert-base-uncased",
    #     model_path='/data/hongbang/CogKTR/datapath/question_answering/NaturalQuestions/raw_data/bert-base-mrc-openqa.pt',
    #     retriever_model_file='/data/hongbang/projects/DPR/downloads/checkpoint/retriever/single-adv-hn/nq/bert-base-encoder.cp',
    #     retriever_index_path='/data/hongbang/projects/DPR/outputs/my_index/',
    #     wiki_passages='/data/hongbang/projects/DPR/downloads/data/wikipedia_split/psgs_w100.tsv',
    #     device=torch.device("cuda:9"),
    #     debug=False,
    # )
    # agent.run()
