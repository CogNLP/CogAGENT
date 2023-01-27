from cogagent.data.datable import DataTable
from cogagent.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import transformers
from cogagent.data.processors.base_processor import BaseProcessor

transformers.logging.set_verbosity_error()


class ModProcessor(BaseProcessor):

    def __init__(self, pretrained_model_name_or_path, max_token_len, addition, do_sample=False):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.max_token_len = max_token_len
        self.addition_dict = addition
        self.do_sample = do_sample
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
        special_tokens_dict = {
            'additional_special_tokens': ['[speaker1]', '[speaker2]']
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)

    def _process(self, data):
        if self.do_sample:
            data = self._do_sample_process(data=data, do_sample=self.do_sample)
        datable = DataTable()
        print("Processing data...")
        for sent, img_id, img_word, neg_img_id, neg_img_word, emotion_id in tqdm(zip(data['sent'],
                                                                                     data['img_id'],
                                                                                     data['img_word'],
                                                                                     data['neg_img_id'],
                                                                                     data['neg_img_word'],
                                                                                     data['emotion_id']),
                                                                                 total=len(data['sent'])):
            tokenized_result = self.tokenizer.encode_plus(text=sent,
                                                          padding="max_length",
                                                          add_special_tokens=True,
                                                          max_length=self.max_token_len,
                                                          truncation=True)
            input_ids = tokenized_result["input_ids"]
            token_type_ids = tokenized_result["token_type_ids"]
            attention_mask = tokenized_result["attention_mask"]
            if all(attention_mask) == 1:
                valid_token_len = self.max_token_len
            else:
                valid_token_len = attention_mask.index(0)

            datable("input_ids", input_ids)
            datable("token_type_ids", token_type_ids)
            datable("attention_mask", attention_mask)
            datable("img_id", img_id)
            datable("img_word", img_word)
            datable("neg_img_id", neg_img_id)
            datable("neg_img_word", neg_img_word)
            datable("emotion_id", emotion_id)
            datable("valid_token_len", valid_token_len)
        datable.add_not2torch("img_word")
        datable.add_not2torch("neg_img_word")
        datable.add_not2torch("img_id")
        datable.add_not2torch("neg_img_id")
        datable.add_not2torch("valid_token_len")
        return DataTableSet(datable)

    def process_train(self, data):
        return self._process(data)

    def process_dev(self, data):
        return self._process(data)

    def process_test(self, data):
        return None

    def get_addition(self):
        self.addition_dict["tokenizer"] = self.tokenizer
        return self.addition_dict


if __name__ == "__main__":
    from cogagent.data.readers.mod_reader import ModReader

    reader = ModReader(raw_data_path="/data/mentianyi/code/CogNLP/datapath/mm_dialog/mod/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    addition = reader.read_addition()

    processor = ModProcessor(
        pretrained_model_name_or_path='YeungNLP/clip-vit-bert-chinese-1M',
        max_token_len=512,
        addition=addition,
        do_sample=True)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    test_dataset = processor.process_test(test_data)
    addition = processor.get_addition()
    print("end")
