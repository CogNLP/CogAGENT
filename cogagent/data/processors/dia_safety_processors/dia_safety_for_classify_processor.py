from cogagent.data.datable import DataTable
from cogagent.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import transformers
from cogagent.data.processors.base_processor import BaseProcessor
from transformers import AutoTokenizer

transformers.logging.set_verbosity_error()  # set transformers logging level


class DiaSafetyForClassify(BaseProcessor):
    def __init__(self, plm, max_token_len, vocab,debug=False):
        super().__init__(debug)
        self.plm = plm
        self.max_token_len = max_token_len
        self.vocab = vocab
        self.tokenizer = AutoTokenizer.from_pretrained(plm)

    def _process(self, data):
        datable = DataTable()
        data = self.debug_process(data)
        print("Processing data...")
        for context,response,category,label,final_label in tqdm(zip(data["context"],data["response"],data["category"],data["label"],data["1vAlabel"]),total=len(data["context"])):
            tokenized_data = self.tokenizer.encode_plus(text=context,
                                                        text_pair=response,
                                                        padding="max_length",
                                                        add_special_tokens=True,
                                                        max_length=self.max_token_len)
            datable("input_ids", tokenized_data["input_ids"])
            datable("attention_mask", tokenized_data["attention_mask"])
            datable("label", final_label)
        return DataTableSet(datable)

    def process_train(self, data):
        return self._process(data)

    def process_dev(self, data):
        return self._process(data)

    def process_test(self, data):
        return self._process(data)


if __name__ == "__main__":
    from cogagent.data.readers.diasafety_reader import DiaSafetyReader

    reader = DiaSafetyReader(raw_data_path="/data/hongbang/CogAGENT/datapath/dialogue_safety/DiaSafety/raw_data")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()

    processor = DiaSafetyForClassify(plm="roberta-base", max_token_len=128, vocab=vocab,debug=False)
    train_dataset = processor.process_train(train_data)
    dev_dataset = processor.process_dev(dev_data)
    test_dataset = processor.process_test(test_data)
