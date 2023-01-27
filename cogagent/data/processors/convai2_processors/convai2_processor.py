from cogagent.data.datable import DataTable
from cogagent.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import transformers
from cogagent.data.processors.base_processor import BaseProcessor
import torch
import numpy as np
import copy

transformers.logging.set_verbosity_error()


class Convai2Processor(BaseProcessor):

    def __init__(self,
                 pretrained_model_name_or_path,
                 max_source_len,
                 max_target_len,
                 addition,
                 do_sample=False):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.do_sample = do_sample
        self.addition = addition
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)

    def _get_tokenized_data(self, data, max_len):
        tokenized_data = self.tokenizer(data,
                                        truncation=True,
                                        padding=True,
                                        max_length=max_len)

        input_ids_data = tokenized_data["input_ids"]
        token_type_ids_data = tokenized_data['token_type_ids']
        attention_mask_data = tokenized_data['attention_mask']

        input_ids_data += [0 for _ in range(max_len - len(input_ids_data))]
        token_type_ids_data += [0 for _ in range(max_len - len(token_type_ids_data))]
        attention_mask_data += [0 for _ in range(max_len - len(attention_mask_data))]

        return [input_ids_data, token_type_ids_data, attention_mask_data]

    def _process(self, data):
        if self.do_sample:
            data = self._do_sample_process(data=data, do_sample=self.do_sample)
        datable = DataTable()
        print("Processing data...")
        for persona, query, response in tqdm(zip(data['persona'],
                                                 data['query'],
                                                 data['response']), total=len(data['persona'])):
            input_ids_persona, token_type_ids_persona, \
            attention_mask_persona = self._get_tokenized_data(data=persona, max_len=self.max_source_len)

            input_ids_query, token_type_ids_query, \
            attention_mask_query = self._get_tokenized_data(data=query, max_len=self.max_source_len)

            input_ids_response, token_type_ids_response, \
            attention_mask_response = self._get_tokenized_data(data=response, max_len=self.max_target_len)

            input_ids = copy.deepcopy(input_ids_persona)
            attention_mask = copy.deepcopy(attention_mask_persona)
            token_type_ids = copy.deepcopy(token_type_ids_persona)
            token_type_ids = token_type_ids * 1
            input_ids.extend(input_ids_query)
            attention_mask.extend(attention_mask_query)
            token_type_ids.extend(token_type_ids_query)

            mask_flag = torch.BoolTensor(1 - np.array(attention_mask_response))
            input_ids_response = torch.tensor(input_ids_response)
            lables = input_ids_response.masked_fill(mask_flag, -100)

            datable("input_ids_persona", input_ids_persona)
            datable("input_ids_query", input_ids_query)

            datable("input_ids", input_ids)
            datable("token_type_ids", token_type_ids)
            datable("attention_mask", attention_mask)

            datable("decoder_input_ids_response", input_ids_response)
            datable("decoder_token_type_ids_response", token_type_ids_response)
            datable("decoder_attention_mask_response", attention_mask_response)
            datable("lables", lables)
        return DataTableSet(datable)

    def process_train(self, data):
        return self._process(data)

    def process_dev(self, data):
        return self._process(data)

    def process_test(self, data):
        return None

    def get_addition(self):
        addition_dict = {}
        addition_dict["pre_positive"] = []
        addition_dict["hyp_positive"] = []
        addition_dict["pre_negative"] = []
        addition_dict["hyp_negative"] = []
        positive_num = len(self.addition["pre_positive"])
        negative_num = len(self.addition["pre_negative"])
        for i in tqdm(range(positive_num)):
            pre_positive = self._get_tokenized_data(data=self.addition["pre_positive"][i], max_len=self.max_source_len)
            hyp_positive = self._get_tokenized_data(data=self.addition["hyp_positive"][i], max_len=self.max_target_len)
            pre_positive[1] = np.array(pre_positive[1]) + 1
            addition_dict["pre_positive"].append(pre_positive)
            addition_dict["hyp_positive"].append(hyp_positive)

        for i in tqdm(range(negative_num)):
            pre_negative = self._get_tokenized_data(data=self.addition["pre_negative"][i], max_len=self.max_source_len)
            hyp_negative = self._get_tokenized_data(data=self.addition["hyp_negative"][i], max_len=self.max_target_len)
            pre_negative[1] = np.array(pre_negative[1]) + 1
            addition_dict["pre_negative"].append(pre_negative)
            addition_dict["hyp_negative"].append(hyp_negative)

        return addition_dict


if __name__ == "__main__":
    from cognlp.data.readers.convai2_reader import Convai2Reader

    reader = Convai2Reader(raw_data_path="/data/mentianyi/code/CogNLP/datapath/controllable_dialog/convai2/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    addition = reader.read_addition()

    processor = Convai2Processor(
        pretrained_model_name_or_path='bert-base-uncased',
        max_source_len=64,
        max_target_len=32,
        do_sample=True,
        addition=addition)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    test_dataset = processor.process_test(test_data)
    addition = processor.get_addition()
    print("end")
