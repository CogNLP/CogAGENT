from cogagent.toolkits.base_toolkit import BaseToolkit
from cogagent.utils.MMCoQA_utils import gen_reader_features, RawResult, to_list, write_predictions, \
    write_final_predictions, tookit_final_predictions
from cogagent.utils.train_utils import move_dict_value_to_device
import pytrec_eval
import torch
import numpy as np
import random
import json
import logging

from cogagent.core.metric.MMCoQA_Metric import MMCoQAMetric
from cogagent.data.readers.MMCoQA_reader import MMCoQAReader
from cogagent.data.processors.MMCoQA_processors.MMCoQA_processor import MMCoQAProcessor, RetrieverInputExample, \
    retriever_convert_example_to_feature
from cogagent.models.MMCoQA.MMCoQA_model import Pipeline, BertForOrconvqaGlobal, BertForRetrieverOnlyPositivePassage
import torch.nn as nn
import torch.optim as optim

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogagent import *
from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

MODEL_CLASSES = {
    'reader': (BertConfig, BertForOrconvqaGlobal, BertTokenizer),
    'retriever': (BertConfig, BertForRetrieverOnlyPositivePassage, BertTokenizer),
}
import torch


class MMCoQAToolkit(BaseToolkit):

    def __init__(self, bert_model=None, model_path=None, vocabulary_path=None, device=None):
        super().__init__()

        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained("/data/yangcheng/MMCoQA/release_test/checkpoint-9185/")
        self.reader_cache_dir = "/data/yangcheng/huggingface_cache/bert-base-uncased/"
        self.reader_config = BertConfig.from_pretrained(
            "/data/yangcheng/MMCoQA/pipline_checkpoints/100test/checkpoint-2500/reader/",
            cache_dir=self.reader_cache_dir if self.reader_cache_dir else None)
        self.reader_config.num_qa_labels = 2
        # this not used for BertForOrconvqaGlobal
        self.reader_config.num_retrieval_labels = 2
        self.reader_config.qa_loss_factor = 1.0
        self.reader_config.retrieval_loss_factor = 1.0
        self.reader_config.proj_size = 128
        self.reader_tokenizer = BertTokenizer.from_pretrained(
            "/data/yangcheng/CogAgent/datapath/huggingface_cache/bert-base-uncased/",
            do_lower_case=True, cache_dir=self.reader_cache_dir if self.reader_cache_dir else None)
        self.reader_model = BertForOrconvqaGlobal.from_pretrained(
            "/data/yangcheng/MMCoQA/pipline_checkpoints/100test/checkpoint-2500/reader/",
            from_tf=bool('.ckpt' in 'bert-base-uncased'), config=self.reader_config,
            cache_dir=self.reader_cache_dir if self.reader_cache_dir else None)
        self.retriever_config_class, retriever_model_class, retriever_tokenizer_class = MODEL_CLASSES['retriever']
        self.retriever_config = BertConfig.from_pretrained(
            "/data/yangcheng/MMCoQA/pipline_checkpoints/100test/checkpoint-2500/retriever/")
        self.retriever_tokenizer = BertTokenizer.from_pretrained(
            "/data/yangcheng/CogAgent/datapath/MMCoQA_data/checkpoint-9185/")
        # self.retriever_model = BertForRetrieverOnlyPositivePassage.from_pretrained(
        #     "/data/yangcheng/MMCoQA/release_test/checkpoint-4000/retriever/", force_download=True)
        self.retriever_model = BertForRetrieverOnlyPositivePassage.from_pretrained(
            "/data/yangcheng/MMCoQA/pipline_checkpoints/100test/checkpoint-2500/retriever/")
        self.model = Pipeline(reader_tokenizer=self.reader_tokenizer)
        self.model.retriever = self.retriever_model
        self.model.reader = self.reader_model
        self.pipline_path = "/data/yangcheng/CogAgent/datapath/MMCoQA_data/experimental_result/MMCoQA_train_test/MMCoQA_test--2023-02-14--03-12-25.54/model/checkpoint-1600/models.pt"
        # self.model.load_state_dict(torch.load(self.pipline_path), False)
        self.model = load_model(self.model, self.pipline_path)
        self.model.retriever.passage_encoder = None
        self.model.retriever.passage_proj = None
        self.model.retriever.image_encoder = None
        self.model.retriever.image_proj = None

        self.top_k_for_retriever = 2000
        self.top_k_for_reader = 5
        self.reader_max_seq_length = 512
        self._query_max_seq_length = 30
        self.predict_dir = "/data/yangcheng/CogAgent/datapath/MMCoQA_data/toolkit_result/"
        self.prefix = '4000'
        self.version_2_with_negative = False
        self.n_best_size = 20
        self.max_answer_length = 40
        self.do_lower_case = True
        self.verbose_logging = False
        self.null_score_diff_threshold = 0.0
        self.use_rerank_prob = True
        self.use_retriever_prob = True

    # def generate(self, sentence):
    #     all_outputs = []
    #     for batch in [sentence[i:] for i in range(0, len(sentence), )]:
    #         input_ids = self.encoder_tokenizer.batch_encode_plus(
    #             batch, max_length=512, padding='max_length', truncation=True, return_tensors="pt",
    #         )["input_ids"]
    #         input_ids = input_ids.to(self.device)
    #         outputs = self.model.generate(
    #             input_ids=input_ids,
    #             num_beams=1,
    #             max_length=128,
    #             length_penalty=2.0,
    #             early_stopping=True,
    #             repetition_penalty=1.0,
    #             do_sample=False,
    #             top_k=10,
    #             top_p=1.0,
    #             num_return_sequences=1,
    #             # temperature=0.7
    #         )
    #         all_outputs.extend(outputs.cpu().numpy())
    #     outputs = [
    #         self.decoder_tokenizer.decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #         for output_id in all_outputs
    #     ]
    #     return outputs

    def generate(self, sentence, batch):
        self.model = self.model.to(self.device)
        qas_id = 'C_393_0'
        orig_question_text = sentence
        question_text_list = []
        question_text_list.append(orig_question_text)
        question_text = str(sentence)
        question_text_for_retriever = orig_question_text

        query_example = RetrieverInputExample(guid=qas_id, text_a=sentence)
        query_feature = retriever_convert_example_to_feature(query_example, self.tokenizer,
                                                             max_length=self._query_max_seq_length)
        # 【1，128】
        query_input_ids = np.asarray(query_feature.input_ids)  # shape 30, dtype int64
        query_token_type_ids = np.asarray(query_feature.token_type_ids)  # shape 30, dtype int64
        query_attention_mask = np.asarray(query_feature.attention_mask)  # shape 30, dtype int64
        qid = qas_id  # str
        answer_text = str('')
        answer_start = 0

        # if entry['question_type'] == "text":
        #     modality_label = [0]
        # elif entry['question_type'] == "table":
        #     modality_label = [1]
        # else:
        #     modality_label = [2]

        # 从ndarray转成Tensor
        query_input_ids = torch.from_numpy(query_input_ids)
        query_token_type_ids = torch.from_numpy(query_token_type_ids)
        query_attention_mask = torch.from_numpy(query_attention_mask)
        answer_start = np.array(answer_start)
        answer_start = torch.from_numpy(answer_start)

        # modality_label = torch.tensor(modality_label)

        # 训练阶段的数据处理
        # self.model.eval()  # we first get query representations in eval mode
        qids = np.asarray(qid).reshape(-1).tolist()
        # qids = qids[0]
        question_texts = np.asarray(question_text).reshape(-1).tolist()
        answer_texts = np.asarray(answer_text).reshape(-1).tolist()
        answer_starts = np.asarray(answer_start).reshape(-1).tolist()
        question_texts = []
        answer_texts = []
        answer_starts = []

        # for i in range(len(batch["question_texts"])):
        question_texts.append(question_text)
        # for i in range(len(batch["answer_texts"])):
        answer_texts.append(answer_text)
        # for i in range(len(batch["answer_starts"])):
        answer_starts.append(0)
        query_input_ids = query_input_ids.unsqueeze(0)
        query_attention_mask = query_attention_mask.unsqueeze(0)
        query_token_type_ids = query_token_type_ids.unsqueeze(0)
        reps_inputs = {'query_input_ids': query_input_ids,
                       'query_attention_mask': query_attention_mask,
                       'query_token_type_ids': query_token_type_ids}
        query_reps = self.model.gen_query_reps(reps_inputs)

        qid_to_idx = batch["qid_to_idx"]
        item_ids = batch["item_ids"]
        item_id_to_idx = batch["item_id_to_idx"]
        item_reps = batch["item_reps"]
        qrels = batch["qrels"]
        qrels_sparse_matrix = batch["qrels_sparse_matrix"]
        gpu_index = batch["gpu_index"]
        itemid_modalities = batch["itemid_modalities"]
        images_titles = batch["images_titles"]

        get_passage_batch = {'itemid_modalities': batch["itemid_modalities"],
                             'passages_dict': batch["passages_dict"],
                             'tables_dict': batch["tables_dict"],
                             'images_dict': batch["images_dict"],
                             'item_ids': batch["item_ids"]}
        #
        retrieval_results = self.model.retrieve(self.top_k_for_retriever, self.top_k_for_reader, qids, qid_to_idx,
                                                query_reps, item_ids, item_id_to_idx, item_reps, qrels,
                                                qrels_sparse_matrix,
                                                gpu_index, get_passage_batch, include_positive_passage=True)
        pids_for_retriever = retrieval_results['pids_for_retriever']
        retriever_probs = retrieval_results['retriever_probs']
        retriever_run_dict, rarank_run_dict = {}, {}
        examples, features = {}, {}
        for i in range(len(qids)):
            retriever_run_dict[qids[i]] = {}
            for j in range(retrieval_results['no_cut_retriever_probs'].shape[1]):
                retriever_run_dict[qids[i]][pids_for_retriever[i, j]] = int(
                    retrieval_results['no_cut_retriever_probs'][i, j])
        pids_for_reader = retrieval_results['pids_for_reader']  # (1,5)
        passages_for_reader = retrieval_results['passages_for_reader']
        labels_for_reader = retrieval_results['labels_for_reader']

        # reader的数据
        reader_batch, batch_examples, batch_features = gen_reader_features(qids, question_texts, answer_texts,
                                                                           answer_starts,
                                                                           pids_for_reader, passages_for_reader,
                                                                           labels_for_reader,
                                                                           self.reader_tokenizer,
                                                                           self.reader_max_seq_length,
                                                                           is_training=False,
                                                                           itemid_modalities=itemid_modalities,
                                                                           item_id_to_idx=item_id_to_idx,
                                                                           images_titles=images_titles)
        retrieval_id2type_dict = []
        retrieval_ids = reader_batch['example_id']
        retrieval_modality_types = reader_batch['item_modality_type'].squeeze(0).squeeze(1).tolist()
        for i in range(5):
            retrieval_id = retrieval_ids[i][8:]
            retrieval_modality_type = retrieval_modality_types[0]
            retrieval_id2type = {"retrieval_id": retrieval_id,
                                 "retrieval_modality_type": retrieval_modality_type}
            retrieval_id2type_dict.append(retrieval_id2type)

        example_ids = reader_batch['example_id']
        examples.update(batch_examples)
        features.update(batch_features)
        reader_batch = {k: v.to(self.device) for k, v in reader_batch.items() if k != 'example_id'}
        retriever_probs = retriever_probs[0]
        retriever_probs = retriever_probs.reshape(-1).tolist()

        inputs = {'input_ids': reader_batch['input_ids'].to(self.device),
                  'attention_mask': reader_batch['input_mask'].to(self.device),
                  'token_type_ids': reader_batch['segment_ids'].to(self.device),
                  'image_input': reader_batch['image_input'].to(self.device),
                  # 'modality_labels': modality_label.to(self.device),
                  'item_modality_type': reader_batch['item_modality_type'].to(self.device),
                  'query_input_ids': query_input_ids.to(self.device),
                  'query_attention_mask': query_attention_mask.to(self.device),
                  'query_token_type_ids': query_token_type_ids.to(self.device)}
        outputs = self.model.reader(**inputs)
        all_results = []
        for i, example_id in enumerate(example_ids):
            result = RawResult(unique_id=example_id,
                               start_logits=to_list(outputs[0][i]),
                               end_logits=to_list(outputs[1][i]),
                               retrieval_logits=to_list(outputs[2][i]),
                               retriever_prob=retriever_probs[i])

            all_results.append(result)
        output_prediction_file = os.path.join(
            self.predict_dir, "instance_predictions_{}.json".format(self.prefix))
        output_nbest_file = os.path.join(
            self.predict_dir, "instance_nbest_predictions_{}.json".format(self.prefix))
        output_final_prediction_file = os.path.join(
            self.predict_dir, "final_predictions_{}.json".format(self.prefix))
        if self.version_2_with_negative:
            output_null_log_odds_file = os.path.join(
                self.predict_dir, "instance_null_odds_{}.json".format(self.prefix))
        else:
            output_null_log_odds_file = None

        all_predictions = write_predictions(examples, features, all_results, self.n_best_size,
                                            self.max_answer_length, self.do_lower_case, output_prediction_file,
                                            output_nbest_file, output_null_log_odds_file, self.verbose_logging,
                                            self.version_2_with_negative, self.null_score_diff_threshold)
        final_ressult = tookit_final_predictions(all_predictions, output_final_prediction_file,
                                                 use_rerank_prob=self.use_rerank_prob,
                                                 use_retriever_prob=self.use_retriever_prob)
        for i in range(5):
            if retrieval_id2type_dict[i]['retrieval_id'] == final_ressult['C_393_0']['idx'][0]:
                retrieval_id2type_dict = {"retrieval_id": final_ressult['C_393_0']['idx'][0],
                                          "retrieval_modality_type": retrieval_id2type_dict[i][
                                              'retrieval_modality_type']}
                break

        reply = final_ressult['C_393_0']['best_span_str'][0]

        retrieval_dict = {}
        if retrieval_id2type_dict['retrieval_modality_type'] == 0:
            retrieval_id = retrieval_id2type_dict['retrieval_id']
            retrieval_type = 'text'
            retrieval_content = batch['passages_dict'][retrieval_id]
            retrieval_dict = {"retrieval_id": retrieval_id,
                              "retrieval_type": retrieval_type,
                              "retrieval_content": retrieval_content}

        if retrieval_id2type_dict['retrieval_modality_type'] == 1:
            retrieval_id = retrieval_id2type_dict['retrieval_id']
            retrieval_type = 'table'
            retrieval_content = batch['tables_dict'][retrieval_id]
            retrieval_dict = {"retrieval_id": retrieval_id,
                              "retrieval_type": retrieval_type,
                              "retrieval_content": retrieval_content,
                              "retrieval_table": batch['raw_tables_dict'][retrieval_id]}

        if retrieval_id2type_dict['retrieval_modality_type'] == 2:
            retrieval_id = retrieval_id2type_dict['retrieval_id']
            retrieval_type = 'image'
            retrieval_image_path = batch['images_dict'][retrieval_id]
            retrieval_content = batch['images_titles'][retrieval_id]
            retrieval_dict = {"retrieval_id": retrieval_id,
                              "retrieval_type": retrieval_type,
                              "retrieval_content": retrieval_content,
                              "retrieval_image_path": retrieval_image_path}
        return reply, retrieval_dict


if __name__ == '__main__':
    from cogagent.toolkits.MMCoQA_toolkit import MMCoQAToolkit
    from PIL import Image

    toolkit = MMCoQAToolkit(device=torch.device("cuda:6"))
    reader = MMCoQAReader(raw_data_path="/data/yangcheng/CogAgent/datapath/MMCoQA_data/data/")
    batch = reader.read_addition()
    sentence = "How many photos of Melbourne make up this collage?"
    print("question:", sentence)
    label, retrieval_dict = toolkit.generate(sentence, batch)

    # filePath = retrieval_dict["retrieval_image_path"]  # 文件路径
    # img = Image.open(filePath)  # 打开图像
    # img.show()  # 显示图像

    print("reply:", label)

# https://github.com/CogNLP/CogAGENT.git
