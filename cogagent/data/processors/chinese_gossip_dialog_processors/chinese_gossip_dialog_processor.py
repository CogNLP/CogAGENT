from cogagent.data.datable import DataTable
from cogagent.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import transformers
from cognlp.data.processors.base_processor import BaseProcessor
import copy

transformers.logging.set_verbosity_error()


class ChineseGossipDialogProcessor(BaseProcessor):

    def __init__(self, pretrained_model_name_or_path, max_token_len, valid_mod, do_sample=False):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.max_token_len = max_token_len
        self.valid_mod = valid_mod
        self.do_sample = do_sample
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.pad_id = self.tokenizer.pad_token_id

    def _process(self, data):
        if self.do_sample:
            data = self._do_sample_process(data=data, do_sample=self.do_sample)
        datable = DataTable()
        print("Processing data...")
        if self.valid_mod == "parallel_valid":
            for utterances in tqdm(data['utterances'], total=len(data['utterances'])):
                input_ids = [self.cls_id]
                for index, utterance in enumerate(utterances):
                    input_ids += self.tokenizer.encode(utterance, add_special_tokens=False)
                    input_ids.append(self.sep_id)
                valid_len = len(input_ids) if len(input_ids) <= self.max_token_len else self.max_token_len
                input_ids += [self.pad_id for _ in range(self.max_token_len - len(input_ids))]
                input_ids = input_ids[:self.max_token_len]
                attention_mask = [1] * valid_len + [0] * (self.max_token_len - valid_len)
                datable("input_ids", input_ids)
                datable("valid_len", valid_len)
                datable("attention_mask", attention_mask)
            datable.add_not2torch("valid_len")
        if self.valid_mod == "serial_valid":
            for utterances in tqdm(data['utterances'], total=len(data['utterances'])):
                input_ids = [self.cls_id]
                for index, utterance in enumerate(utterances):
                    if index > 0:
                        current_input_ids = copy.deepcopy(input_ids)
                        current_input_ids = current_input_ids[:self.max_token_len]
                        # current_device_flag用于在当前
                        datable("current_device_flag", 0)
                        datable("input_ids", current_input_ids)
                        datable("labels", utterance)
                    input_ids += self.tokenizer.encode(utterance, add_special_tokens=False)
                    input_ids.append(self.sep_id)
            datable.add_not2torch("input_ids")
        return DataTableSet(datable)

    def process_train(self, data):
        if self.do_sample:
            data = self._do_sample_process(data=data, do_sample=self.do_sample)
        datable = DataTable()
        print("Processing data...")
        for utterances in tqdm(data['utterances'], total=len(data['utterances'])):
            input_ids = [self.cls_id]
            for utterance in utterances:
                input_ids += self.tokenizer.encode(utterance, add_special_tokens=False)
                input_ids.append(self.sep_id)
            valid_len = len(input_ids) if len(input_ids) <= self.max_token_len else self.max_token_len
            input_ids += [self.pad_id for _ in range(self.max_token_len - len(input_ids))]
            input_ids = input_ids[:self.max_token_len]
            attention_mask = [1] * valid_len + [0] * (self.max_token_len - valid_len)
            datable("input_ids", input_ids)
            datable("valid_len", valid_len)
            datable("attention_mask", attention_mask)
        datable.add_not2torch("valid_len")
        return DataTableSet(datable)

    def process_dev(self, data):
        return self._process(data)

    def process_test(self, data):
        return self._process(data)

    def get_addition(self):
        addition_dict = {}
        addition_dict["cls_id"] = self.cls_id
        addition_dict["sep_id"] = self.sep_id
        addition_dict["pad_id"] = self.pad_id
        addition_dict["valid_mod"] = self.valid_mod
        addition_dict["tokenizer"] = self.tokenizer
        return addition_dict


if __name__ == "__main__":
    from cognlp.data.readers.chinese_gossip_dialog_reader import ChineseGossipDialogReader

    reader = ChineseGossipDialogReader(
        raw_data_path="/data/mentianyi/code/CogNLP/datapath/gossip_dialog/chinese_gossip_dialog/raw_data")
    train_data, dev_data, test_data = reader.read_all()

    processor = ChineseGossipDialogProcessor(
        pretrained_model_name_or_path="uer/gpt2-chinese-cluecorpussmall",
        max_token_len=512,
        valid_mod="serial_valid",
        do_sample=True)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    test_dataset = processor.process_test(test_data)
    addition = processor.get_addition()
    print("end")
