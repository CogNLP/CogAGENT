import scipy as sp
import torch
import os
from cogagent.data.datable import DataTable
from cogagent.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
import transformers
import json
from cogagent.data.processors.base_processor import BaseProcessor
from cogagent.models.MMCoQA.MMCoQA_model import Pipeline, BertForOrconvqaGlobal, BertForRetrieverOnlyPositivePassage
from cogagent.utils.MMCoQA_utils import (LazyQuacDatasetGlobal, RawResult, write_predictions, write_final_predictions,
                                         get_retrieval_metrics, gen_reader_features)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"


# transformers.logging.set_verbosity_error()  # set transformers logging level


class MMCoQAProcessor(BaseProcessor):
    def __init__(self,):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("/data/yangcheng/MMCoQA/release_test/checkpoint-9185/")
        self._query_max_seq_length = 30
        self._history_num = 1
        self._prepend_history_questions = True
        self._include_first_for_retriever = True
        self._prepend_history_answers = False

    def _process(self, data):
        datable = DataTable()
        print("Processing data...")
        for sentence, label in tqdm(zip(data['sentence'], data['label']), total=len(data['sentence'])):
            # token = self.tokenizer.encode(text=sentence, truncation=True, padding="max_length", add_special_tokens=True,
            #                               max_length=self.max_token_len)
            # datable("input_ids", token)
            tokenized_data = self.tokenizer.encode_plus(text=sentence,
                                                        padding="max_length",
                                                        add_special_tokens=True,
                                                        max_length=self.max_token_len)
            datable("input_ids", tokenized_data["input_ids"])
            datable("token_type_ids", tokenized_data["token_type_ids"])
            datable("attention_mask", tokenized_data["attention_mask"])
            datable("label", self.vocab["label_vocab"].label2id(label))
        return DataTableSet(datable)

    def process_train(self, train_data, batch):
        datable = DataTable()
        print("Processing train data...")
        train_data_length = len(train_data)
        for i in range(train_data_length):
        # for i in range(50):
            # dataset阶段数据处理
            line = train_data[i]
            entry = json.loads(line.strip())
            qas_id = entry["qid"]
            # question_text_for_retriever = entry["gold_question"]

            # orig_question_text = entry["question"]
            # history = entry['history']
            # question_text_list = []
            # if self._history_num > 0:
            #     for turn in history[- self._history_num:]:
            #         if self._prepend_history_questions:
            #             question_text_list.append(turn['question'])
            #         if self._prepend_history_answers:
            #             question_text_list.append(turn['answer'][0]['answer'])
            # question_text_list.append(orig_question_text)
            # question_text = ' [SEP] '.join(question_text_list)
            # question_text_for_retriever = question_text
            #
            # # include the first question in addition to history_num for retriever (not reader)
            # if self._include_first_for_retriever and len(history) > 0:
            #     first_question = history[0]['question']
            #     if first_question != question_text_list[0]:
            #         question_text_for_retriever = first_question + ' [SEP] ' + question_text

            question_text = entry["gold_question"]
            question_text_for_retriever = entry["gold_question"]
            query_example = RetrieverInputExample(guid=qas_id, text_a=question_text_for_retriever)
            query_feature = retriever_convert_example_to_feature(query_example, self.tokenizer,
                                                                 max_length=self._query_max_seq_length)
            # 【1，128】
            query_input_ids = np.asarray(query_feature.input_ids)  # shape 30, dtype int64
            query_token_type_ids = np.asarray(query_feature.token_type_ids)  # shape 30, dtype int64
            query_attention_mask = np.asarray(query_feature.attention_mask)  # shape 30, dtype int64
            qid = qas_id  # str
            answer_text = str(entry['answer'][0]['answer'])
            answer_start = 0

            if entry['question_type'] == "text":
                modality_label = [0]
            elif entry['question_type'] == "table":
                modality_label = [1]
            else:
                modality_label = [2]

            # 从ndarray转成Tensor
            query_input_ids = torch.from_numpy(query_input_ids)
            query_token_type_ids = torch.from_numpy(query_token_type_ids)
            query_attention_mask = torch.from_numpy(query_attention_mask)
            answer_start = np.array(answer_start)
            answer_start = torch.from_numpy(answer_start)

            modality_label = torch.tensor(modality_label)

            qids = np.asarray(qid).reshape(-1).tolist()
            qids = qids[0]
            question_texts = np.asarray(question_text).reshape(-1).tolist()
            answer_texts = np.asarray(answer_text).reshape(-1).tolist()
            answer_starts = np.asarray(answer_start).reshape(-1).tolist()

            # reps_inputs = {'query_input_ids': query_input_ids.unsqueeze(0),
            #                'query_attention_mask': query_attention_mask.unsqueeze(0),
            #                'query_token_type_ids': query_token_type_ids.unsqueeze(0)}
            #
            # query_reps = gen_query_reps(self.model, reps_inputs, device)

            # get_passage_batch = {'itemid_modalities': batch["itemid_modalities"],
            #                      'passages_dict': batch["passages_dict"],
            #                      'tables_dict': batch["tables_dict"],
            #                      'images_dict': batch["images_dict"],
            #                      'item_ids': batch["item_ids"]}
            #
            # retrieval_results = retrieve(self.top_k_for_retriever, self.top_k_for_reader, qids, qid_to_idx,
            #                              query_reps, item_ids, item_id_to_idx, item_reps, qrels, qrels_sparse_matrix,
            #                              gpu_index, get_passage_batch, include_positive_passage=True)
            # passage_reps_for_retriever = retrieval_results['passage_reps_for_retriever']  # (1,2000,128) float32
            # labels_for_retriever = retrieval_results['labels_for_retriever']  # (1,2000) int64
            # pids_for_reader = retrieval_results['pids_for_reader']  # (1,5)
            # passages_for_reader = retrieval_results['passages_for_reader']
            # passages_for_reader = np.expand_dims(passages_for_reader, 0)  # 把原来size 5 增加一个维度 变成（1,5）
            # labels_for_reader = retrieval_results['labels_for_reader']
            # # reader的数据
            # reader_batch = gen_reader_features(qids, question_texts, answer_texts, answer_starts,
            #                                    pids_for_reader, passages_for_reader, labels_for_reader,
            #                                    self.reader_tokenizer, self.reader_max_seq_length, is_training=True,
            #                                    itemid_modalities=batch["itemid_modalities"],
            #                                    item_id_to_idx=item_id_to_idx,
            #                                    images_titles=batch["images_titles"])
            # # retriever的模型输入
            # # inputs = {'query_input_ids': batch['query_input_ids'].to(args.device),
            # #           'query_attention_mask': batch['query_attention_mask'].to(args.device),
            # #           'query_token_type_ids': batch['query_token_type_ids'].to(args.device),
            # #           'passage_rep': torch.from_numpy(passage_reps_for_retriever).to(args.device),
            # #           'retrieval_label': torch.from_numpy(labels_for_retriever).to(args.device)}
            # # reader的模型输入
            # # inputs = {'input_ids': reader_batch['input_ids'],
            # #           'attention_mask': reader_batch['input_mask'],
            # #           'token_type_ids': reader_batch['segment_ids'],
            # #           'start_positions': reader_batch['start_position'],
            # #           'end_positions': reader_batch['end_position'],
            # #           'retrieval_label': reader_batch['retrieval_label'],
            # #           'image_input': reader_batch['image_input'],
            # #           'modality_labels': batch['modality_label'].to(args.device),
            # #           'item_modality_type': reader_batch['item_modality_type'],
            # #           'query_input_ids': batch['query_input_ids'].to(args.device),
            # #           'query_attention_mask': batch['query_attention_mask'].to(args.device),
            # #           'query_token_type_ids': batch['query_token_type_ids'].to(args.device)}
            #
            # passage_rep = torch.from_numpy(passage_reps_for_retriever)
            # retriever_label = torch.from_numpy(labels_for_retriever)
            datable("images_titles", batch["images_titles"])
            datable("qid_to_idx", batch["qid_to_idx"])
            datable("item_ids", batch["item_ids"])
            datable("item_id_to_idx", batch["item_id_to_idx"])
            datable("item_reps", batch["item_reps"])
            datable("qrels", batch["qrels"])
            datable("qrels_sparse_matrix", batch["qrels_sparse_matrix"])
            datable("gpu_index", batch["gpu_index"])
            datable("itemid_modalities", batch["itemid_modalities"])
            datable("passages_dict", batch["passages_dict"])
            datable("tables_dict", batch["tables_dict"])
            datable("images_dict", batch["images_dict"])
            datable("qids", qids)
            datable("question_texts", question_texts)
            datable("answer_texts", answer_texts)
            datable("answer_starts", answer_starts)
            datable("train_query_input_ids", query_input_ids)  # shape 30, dtype int64
            datable("train_query_attention_mask", query_attention_mask)  # shape 30, dtype int64
            datable("train_query_token_type_ids", query_token_type_ids)  # shape 30, dtype int64
            # datable("train_passage_rep", passage_rep.squeeze())  # torch.size [2000,128], dtype torch.float32
            # datable("train_retriever_label", retriever_label.squeeze())  # torch.size [2000], dtype torch.int64
            # datable("train_reader_input_ids",
            #         reader_batch['input_ids'].squeeze())  # torch.size [5,512], dtype torch.int64
            # datable("train_reader_attention_mask",
            #         reader_batch['input_mask'].squeeze())  # torch.size [5,512], dtype torch.int64
            # datable("train_reader_token_type_ids",
            #         reader_batch['segment_ids'].squeeze())  # torch.size [5,512], dtype torch.int64
            # datable("train_reader_start_positions",
            #         reader_batch['start_position'].squeeze(0))  # torch.size [5,1], dtype torch.int64
            # datable("train_reader_end_positions",
            #         reader_batch['end_position'].squeeze(0))  # torch.size [5,1], dtype torch.int64
            # datable("train_reader_retrieval_label",
            #         reader_batch['retrieval_label'].squeeze(0))  # torch.size [5,1], dtype torch.int64
            # datable("train_reader_image_input",
            #         reader_batch['image_input'].squeeze())  # torch.size [15,512,512], dtype torch.float64
            # datable("train_reader_item_modality_type",
            #         reader_batch['item_modality_type'].squeeze(0))  # torch.size [5,1], dtype torch.int64
            datable("train_modality_labels", modality_label)  # int

        return DataTableSet(datable)

    def process_dev(self, dev_data, batch):
        datable = DataTable()
        print("Processing dev data...")
        train_data_length = len(dev_data)
        for i in range(train_data_length):
        # for i in range(50):
            # dataset阶段数据处理
            line = dev_data[i]
            entry = json.loads(line.strip())
            qas_id = entry["qid"]
            # question_text_for_retriever = entry["gold_question"]

            # orig_question_text = entry["question"]
            # history = entry['history']
            # question_text_list = []
            # if self._history_num > 0:
            #     for turn in history[- self._history_num:]:
            #         if self._prepend_history_questions:
            #             question_text_list.append(turn['question'])
            #         if self._prepend_history_answers:
            #             question_text_list.append(turn['answer'][0]['answer'])
            # question_text_list.append(orig_question_text)
            # question_text = ' [SEP] '.join(question_text_list)
            # question_text_for_retriever = question_text
            #
            # # include the first question in addition to history_num for retriever (not reader)
            # if self._include_first_for_retriever and len(history) > 0:
            #     first_question = history[0]['question']
            #     if first_question != question_text_list[0]:
            #         question_text_for_retriever = first_question + ' [SEP] ' + question_text

            question_text = entry["gold_question"]
            question_text_for_retriever = entry["gold_question"]
            query_example = RetrieverInputExample(guid=qas_id, text_a=question_text_for_retriever)
            query_feature = retriever_convert_example_to_feature(query_example, self.tokenizer,
                                                                 max_length=self._query_max_seq_length)
            # 【1，128】
            query_input_ids = np.asarray(query_feature.input_ids)  # shape 30, dtype int64
            query_token_type_ids = np.asarray(query_feature.token_type_ids)  # shape 30, dtype int64
            query_attention_mask = np.asarray(query_feature.attention_mask)  # shape 30, dtype int64
            qid = qas_id  # str
            answer_text = str(entry['answer'][0]['answer'])
            answer_start = 0

            if entry['question_type'] == "text":
                modality_label = [0]
            elif entry['question_type'] == "table":
                modality_label = [1]
            else:
                modality_label = [2]

            # 从ndarray转成Tensor
            query_input_ids = torch.from_numpy(query_input_ids)
            query_token_type_ids = torch.from_numpy(query_token_type_ids)
            query_attention_mask = torch.from_numpy(query_attention_mask)
            answer_start = np.array(answer_start)
            answer_start = torch.from_numpy(answer_start)

            modality_label = torch.tensor(modality_label)

            # 训练阶段的数据处理
            # self.model.eval()  # we first get query representations in eval mode
            qids = np.asarray(qid).reshape(-1).tolist()
            qids = qids[0]
            question_texts = np.asarray(question_text).reshape(-1).tolist()
            answer_texts = np.asarray(answer_text).reshape(-1).tolist()
            answer_starts = np.asarray(answer_start).reshape(-1).tolist()

            datable("qids", qids)
            datable("question_texts", question_texts)
            datable("answer_texts", answer_texts)
            datable("answer_starts", answer_starts)
            datable("modality_label", modality_label)
            # reps_inputs = {'query_input_ids': query_input_ids.unsqueeze(0),
            #                'query_attention_mask': query_attention_mask.unsqueeze(0),
            #                'query_token_type_ids': query_token_type_ids.unsqueeze(0)}

            # query_reps = gen_query_reps(self.model, reps_inputs, device)
            # qid_to_idx = batch["qid_to_idx"]
            # item_ids = batch["item_ids"]
            # item_id_to_idx = batch["item_id_to_idx"]
            # item_reps = batch["item_reps"]
            # qrels = batch["qrels"]
            # qrels_sparse_matrix = batch["qrels_sparse_matrix"]
            # gpu_index = batch["gpu_index"]

            # get_passage_batch = {'itemid_modalities': batch["itemid_modalities"],
            #                      'passages_dict': batch["passages_dict"],
            #                      'tables_dict': batch["tables_dict"],
            #                      'images_dict': batch["images_dict"],
            #                      'item_ids': batch["item_ids"]}

            # retrieval_results = retrieve(self.top_k_for_retriever, self.top_k_for_reader, qids, qid_to_idx,
            #                              query_reps, item_ids, item_id_to_idx, item_reps, qrels, qrels_sparse_matrix,
            #                              gpu_index, get_passage_batch, include_positive_passage=True)
            # pids_for_retriever = retrieval_results['pids_for_retriever']
            # retriever_probs = retrieval_results['retriever_probs']
            # retriever_run_dict, rarank_run_dict = {}, {}
            # examples, features = {}, {}
            # for i in range(len(qids)):
            #     retriever_run_dict[qids[i]] = {}
            #     for j in range(retrieval_results['no_cut_retriever_probs'].shape[1]):
            #         retriever_run_dict[qids[i]][pids_for_retriever[i, j]] = int(
            #             retrieval_results['no_cut_retriever_probs'][i, j])
            # datable("dev_retriever_run_dict", retriever_run_dict)
            # pids_for_reader = retrieval_results['pids_for_reader']  # (1,5)
            # passages_for_reader = retrieval_results['passages_for_reader']
            # passages_for_reader = np.expand_dims(passages_for_reader, 0)  # 把原来size 5 增加一个维度 变成（1,5）
            # labels_for_reader = retrieval_results['labels_for_reader']
            #
            # # reader的数据
            # reader_batch, batch_examples, batch_features = gen_reader_features(qids, question_texts, answer_texts,
            #                                                                    answer_starts,
            #                                                                    pids_for_reader, passages_for_reader,
            #                                                                    labels_for_reader,
            #                                                                    self.reader_tokenizer,
            #                                                                    self.reader_max_seq_length,
            #                                                                    is_training=False,
            #                                                                    itemid_modalities=batch[
            #                                                                        "itemid_modalities"],
            #                                                                    item_id_to_idx=item_id_to_idx,
            #                                                                    images_titles=batch["images_titles"])
            # example_ids = reader_batch['example_id']
            # examples.update(batch_examples)
            # features.update(batch_features)
            # reader_batch = {k: v.to(device) for k, v in reader_batch.items() if k != 'example_id'}
            # retriever_probs = retriever_probs[0]
            # retriever_probs = retriever_probs.reshape(-1).tolist()

            # datable("dev_examples", examples)
            # datable("dev_features", features)
            # datable("dev_example_ids", example_ids)
            # datable("dev_retriever_probs", retriever_probs)
            # # datable("dev_retriever_run_dict", retriever_run_dict)
            # # torch.size [5,512], dtype torch.int64
            # datable("dev_reader_input_ids", reader_batch['input_ids'].squeeze())
            # # torch.size [5,512], dtype torch.int64
            # datable("dev_reader_attention_mask", reader_batch['input_mask'].squeeze())
            # # torch.size [5,512], dtype torch.int64
            # datable("dev_reader_token_type_ids", reader_batch['segment_ids'].squeeze())
            # # torch.size [15,512,512], dtype torch.float64
            # datable("dev_reader_image_input", reader_batch['image_input'].squeeze())
            datable("dev_modality_labels", modality_label)  # int
            # # torch.size [5,1], dtype torch.int64
            # datable("dev_reader_item_modality_type", reader_batch['item_modality_type'].squeeze(0))
            datable("qid_to_idx", batch["qid_to_idx"])
            datable("item_ids", batch["item_ids"])
            datable("item_id_to_idx", batch["item_id_to_idx"])
            datable("item_reps", batch["item_reps"])
            datable("qrels", batch["qrels"])
            datable("qrels_sparse_matrix", batch["qrels_sparse_matrix"])
            datable("itemid_modalities", batch["itemid_modalities"])
            datable("passages_dict", batch["passages_dict"])
            datable("tables_dict", batch["tables_dict"])
            datable("images_dict", batch["images_dict"])
            datable("gpu_index", batch["gpu_index"])
            datable("images_titles", batch["images_titles"])
            datable("dev_query_input_ids", query_input_ids)  # shape 30, dtype int64
            datable("dev_query_attention_mask", query_attention_mask)  # shape 30, dtype int64
            datable("dev_query_token_type_ids", query_token_type_ids)  # shape 30, dtype int64

        return DataTableSet(datable)

    def process_test(self, data, batch, device):
        datable = DataTable()
        print("Processing test data...")
        train_data_length = len(dev_data)
        # for i in range(train_data_length):
        for i in range(50):
            # dataset阶段数据处理
            line = dev_data[i]
            entry = json.loads(line.strip())
            qas_id = entry["qid"]
            orig_question_text = entry["question"]
            history = entry['history']
            question_text_list = []
            if self._history_num > 0:
                for turn in history[- self._history_num:]:
                    if self._prepend_history_questions:
                        question_text_list.append(turn['question'])
                    if self._prepend_history_answers:
                        question_text_list.append(turn['answer'][0]['answer'])
            question_text_list.append(orig_question_text)
            question_text = ' [SEP] '.join(question_text_list)
            question_text_for_retriever = question_text
            if self._include_first_for_retriever and len(history) > 0:
                first_question = history[0]['question']
                if first_question != question_text_list[0]:
                    question_text_for_retriever = first_question + ' [SEP] ' + question_text
            query_example = RetrieverInputExample(guid=qas_id, text_a=question_text_for_retriever)
            query_feature = retriever_convert_example_to_feature(query_example, self.tokenizer,
                                                                 max_length=self._query_max_seq_length)
            # 【1，128】
            query_input_ids = np.asarray(query_feature.input_ids)  # shape 30, dtype int64
            query_token_type_ids = np.asarray(query_feature.token_type_ids)  # shape 30, dtype int64
            query_attention_mask = np.asarray(query_feature.attention_mask)  # shape 30, dtype int64
            qid = qas_id  # str
            answer_text = str(entry['answer'][0]['answer'])
            answer_start = 0

            if entry['question_type'] == "text":
                modality_label = [0]
            elif entry['question_type'] == "table":
                modality_label = [1]
            else:
                modality_label = [2]

            # 从ndarray转成Tensor
            query_input_ids = torch.from_numpy(query_input_ids)
            query_token_type_ids = torch.from_numpy(query_token_type_ids)
            query_attention_mask = torch.from_numpy(query_attention_mask)
            answer_start = np.array(answer_start)
            answer_start = torch.from_numpy(answer_start)

            modality_label = torch.tensor(modality_label)

            qids = np.asarray(qid).reshape(-1).tolist()
            qids = qids[0]
            question_texts = np.asarray(question_text).reshape(-1).tolist()
            answer_texts = np.asarray(answer_text).reshape(-1).tolist()
            answer_starts = np.asarray(answer_start).reshape(-1).tolist()

            datable("qids", qids)
            datable("question_texts", question_texts)
            datable("answer_texts", answer_texts)
            datable("answer_starts", answer_starts)
            datable("modality_label", modality_label)
            datable("qid_to_idx", batch["qid_to_idx"])
            datable("item_ids", batch["item_ids"])
            datable("item_id_to_idx", batch["item_id_to_idx"])
            datable("item_reps", batch["item_reps"])
            datable("qrels", batch["qrels"])
            datable("qrels_sparse_matrix", batch["qrels_sparse_matrix"])
            datable("itemid_modalities", batch["itemid_modalities"])
            datable("passages_dict", batch["passages_dict"])
            datable("tables_dict", batch["tables_dict"])
            datable("images_dict", batch["images_dict"])
            datable("gpu_index", batch["gpu_index"])
            datable("images_titles", batch["images_titles"])
            datable("test_query_input_ids", query_input_ids)  # shape 30, dtype int64
            datable("test_query_attention_mask", query_attention_mask)  # shape 30, dtype int64
            datable("test_query_token_type_ids", query_token_type_ids)  # shape 30, dtype int64
        return DataTableSet(datable)


def retriever_convert_example_to_feature(example, tokenizer,
                                         max_length=512,
                                         pad_on_left=False,
                                         pad_token=0,
                                         pad_token_segment_id=0,
                                         mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """

    inputs = tokenizer.encode_plus(
        example.text_a,
        example.text_b,
        add_special_tokens=True,
        max_length=max_length,
    )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                        max_length)
    assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                        max_length)

    if False:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        logger.info("label: %s" % (example.label))

    feature = RetrieverInputFeatures(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     label=example.label)

    return feature


class RetrieverInputExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class RetrieverInputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def gen_query_reps(model, batch, device):
    model.eval()
    # batch = {k: v.to("cuda:0") for k, v in batch.items()
             # if k not in ['example_id', 'qid', 'question_text', 'answer_text', 'answer_start']}
    with torch.no_grad():
        # model = model.to("cuda:0")
        inputs = {}
        inputs['query_input_ids'] = batch['query_input_ids']
        inputs['query_attention_mask'] = batch['query_attention_mask']
        inputs['query_token_type_ids'] = batch['query_token_type_ids']
        outputs = model.retriever(**inputs)
        query_reps = outputs[0]

    return query_reps


def retrieve(top_k_for_retriever, top_k_for_reader, qids, qid_to_idx, query_reps,
             item_ids, item_id_to_idx, item_reps, qrels, qrels_sparse_matrix,
             gpu_index, batch, include_positive_passage=False):
    query_reps = query_reps.detach().cpu().numpy()
    D, I = gpu_index.search(query_reps, top_k_for_retriever)

    pidx_for_retriever = np.copy(I)
    qidx = [qid_to_idx[qid] for qid in qids]
    qidx_expanded = np.expand_dims(qidx, axis=1)
    qidx_expanded = np.repeat(qidx_expanded, top_k_for_retriever, axis=1)
    labels_for_retriever = qrels_sparse_matrix[qidx_expanded, pidx_for_retriever].toarray()
    # print('labels_for_retriever before', labels_for_retriever)
    if include_positive_passage:
        for i, (qid, labels_per_query) in enumerate(zip(qids, labels_for_retriever)):
            has_positive = np.sum(labels_per_query)
            if not has_positive:
                positive_pid = list(qrels[qid].keys())[0]
                positive_pidx = item_id_to_idx[positive_pid]
                pidx_for_retriever[i][-1] = positive_pidx
        labels_for_retriever = qrels_sparse_matrix[qidx_expanded, pidx_for_retriever].toarray()
        # print('labels_for_retriever after', labels_for_retriever)
        assert np.sum(labels_for_retriever) >= len(labels_for_retriever)
    pids_for_retriever = item_ids[pidx_for_retriever]
    passage_reps_for_retriever = item_reps[pidx_for_retriever]

    scores = D[:, :top_k_for_reader]
    retriever_probs = sp.special.softmax(scores, axis=1)
    pidx_for_reader = I[:, :top_k_for_reader]
    # print('pidx_for_reader', pidx_for_reader)
    # print('qids', qids)
    # print('qidx', qidx)
    qidx_expanded = np.expand_dims(qidx, axis=1)
    qidx_expanded = np.repeat(qidx_expanded, top_k_for_reader, axis=1)
    # print('qidx_expanded', qidx_expanded)

    labels_for_reader = qrels_sparse_matrix[qidx_expanded, pidx_for_reader].toarray()
    # print('labels_for_reader before', labels_for_reader)
    # print('labels_for_reader before', labels_for_reader)
    if include_positive_passage:
        for i, (qid, labels_per_query) in enumerate(zip(qids, labels_for_reader)):
            has_positive = np.sum(labels_per_query)
            if not has_positive:
                positive_pid = list(qrels[qid].keys())[0]
                positive_pidx = item_id_to_idx[positive_pid]
                pidx_for_reader[i][-1] = positive_pidx
        labels_for_reader = qrels_sparse_matrix[qidx_expanded, pidx_for_reader].toarray()
        # print('labels_for_reader after', labels_for_reader)
        assert np.sum(labels_for_reader) >= len(labels_for_reader)
    # print('labels_for_reader after', labels_for_reader)
    pids_for_reader = item_ids[pidx_for_reader]
    # print('pids_for_reader', pids_for_reader)
    passages_for_reader = get_passages(pidx_for_reader, batch)
    # passages_for_reader = np.vectorize(get_passage(pidx_for_reader, batch))
    # we do not need to modify scores and probs matrices because they will only be
    # needed at evaluation, where include_positive_passage will be false

    return {'qidx': qidx,
            'pidx_for_retriever': pidx_for_retriever,
            'pids_for_retriever': pids_for_retriever,
            'passage_reps_for_retriever': passage_reps_for_retriever,
            'labels_for_retriever': labels_for_retriever,
            'retriever_probs': retriever_probs,
            'pidx_for_reader': pidx_for_reader,
            'pids_for_reader': pids_for_reader,
            'passages_for_reader': passages_for_reader,
            'labels_for_reader': labels_for_reader,
            'no_cut_retriever_probs': D}


def get_passages(ids, batch):
    ids = list(np.squeeze(ids))
    item_contexts = []
    for i in ids:
        itemid_modalities = batch["itemid_modalities"]
        passages_dict = batch["passages_dict"]
        tables_dict = batch["tables_dict"]
        images_dict = batch["images_dict"]
        item_ids = batch["item_ids"]
        if itemid_modalities[i] == 'text':
            item_context = passages_dict[item_ids[i]]
        elif itemid_modalities[i] == 'table':
            item_context = tables_dict[item_ids[i]]
        elif itemid_modalities[i] == 'image':
            item_context = images_dict[item_ids[i]]
        item_contexts.append(item_context)

    return np.array(item_contexts)


if __name__ == "__main__":
    from cogagent.data.readers.MMCoQA_reader import MMCoQAReader

    reader = MMCoQAReader(raw_data_path="/data/yangcheng/CogAgent/datapath/MMCoQA_data/data/")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()

    processor = MMCoQAProcessor()
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    test_dataset = processor.process_test(test_data)
    print("end")
