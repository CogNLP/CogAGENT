from cogagent.data.datable import DataTable
from cogagent.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import transformers
import os
import torch
import pandas as pd
from cogagent.data.processors.base_processor import BaseProcessor
from cogagent.utils.KE_Blender_utils import Seq2SeqDataset, SimpleSummarizationDataset
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
    BlenderbotConfig,
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
    BlenderbotSmallConfig,
    # BlenderbotSmallForConditionalGeneration,
    BlenderbotSmallTokenizer,
    BertConfig,
    BertForMaskedLM,
    BertModel,
    BertTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)
from simpletransformers.config.global_args import global_args
from simpletransformers.config.model_args import Seq2SeqArgs

transformers.logging.set_verbosity_error()  # set transformers logging level

class KE_BlenderProcessor(BaseProcessor):
    def __init__(self, plm, vocab):
        super().__init__()
        self.plm = plm
        # self.max_token_len = max_token_len
        self.vocab = vocab
        self.tokenizer = BertTokenizer.from_pretrained(plm)
        self.config_class = BlenderbotSmallConfig
        self.model_class = BlenderbotForConditionalGeneration
        self.tokenizer_class = BlenderbotSmallTokenizer
        self.encoder_decoder_type = "blender"
        self.encoder_decoder_name = "facebook/blenderbot_small-90M"
        self.encoder_tokenizer = self.tokenizer_class.from_pretrained(self.encoder_decoder_name, additional_special_tokens=['__defi__', '__hype__', '[MASK]'])
        self.decoder_tokenizer = self.encoder_tokenizer
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        args = {
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "max_seq_length": 512,
            "train_batch_size": 16,
            "num_train_epochs": 3,
            "save_eval_checkpoints": False,
            "save_model_every_epoch": True,
            "evaluate_during_training": True,
            "evaluate_generated_text": True,
            "evaluate_during_training_verbose": True,
            "use_multiprocessing": False,
            "max_length": 128,
            "manual_seed": 42,
            "n_gpu": 1,
            "gradient_accumulation_steps": 4,
            "output_dir": "/KE_Blender",
            "model_name": "facebook/blenderbot_small-90M",
            "model_type": "blender"
        }

        self.args = self._load_model_args(self.encoder_decoder_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, Seq2SeqArgs):
            self.args = args

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

    def process_train(self, data):
        datable = DataTable()
        print("Processing train data...")
        mode = "train"
        train_data = pd.DataFrame(data, columns=["input_text", "target_text"])
        train_dataset = self.load_and_cache_examples(train_data, verbose=True)
        # for i in range(500):
        for i in range(len(train_dataset)):
            train_input = train_dataset[i]
            train_inputs = self._get_inputs_dict(train_input)

            train_input_ids = train_inputs["input_ids"].squeeze()
            train_attention_mask = train_inputs["attention_mask"].squeeze()
            train_decoder_input_ids = train_inputs["decoder_input_ids"].squeeze()
            train_labels = train_inputs["labels"].squeeze()
            datable("train_input_ids", train_input_ids)
            datable("train_attention_mask", train_attention_mask)
            datable("train_decoder_input_ids", train_decoder_input_ids)
            datable("train_labels", train_labels)
            datable("train_inputs", train_inputs)
        return DataTableSet(datable)

    def process_dev(self, data):
        datable = DataTable()
        print("Processing valid data...")
        mode = "dev"
        # for i in range(len(data)):
        num = len(data)
        dev_data = pd.DataFrame(data, columns=["input_text", "target_text"])
        dev_dataset = self.load_and_cache_examples(dev_data, verbose=True)
        # dev_input = ['source_ids', 'source_mask', 'target_ids']
        for i in range(len(dev_dataset)):
            dev_input = dev_dataset[i]
            dev_inputs = self._get_inputs_dict(dev_input)
            datable("dev_text", data[i][0])
            datable("dev_labels_text", data[i][1])
            dev_input_ids = dev_inputs["input_ids"].squeeze()
            dev_attention_mask = dev_inputs["attention_mask"].squeeze()
            dev_decoder_input_ids = dev_inputs["decoder_input_ids"].squeeze()
            dev_labels = dev_inputs["labels"].squeeze()
            datable("dev_input_ids", dev_input_ids)
            datable("dev_attention_mask", dev_attention_mask)
            datable("dev_decoder_input_ids", dev_decoder_input_ids)
            datable("dev_labels", dev_labels)
            datable("dev_inputs", dev_inputs)

        return DataTableSet(datable)

    def process_test(self, data):
        datable = DataTable()
        print("Processing test data...")
        mode = "train"
        test_data = pd.DataFrame(data, columns=["input_text", "target_text"])
        test_dataset = self.load_and_cache_examples(test_data, verbose=True)
        for i in range(len(test_dataset)):
            test_input = test_dataset[i]
            test_inputs = self._get_inputs_dict(test_input)

            test_input_ids = test_inputs["input_ids"].squeeze()
            test_attention_mask = test_inputs["attention_mask"].squeeze()
            test_decoder_input_ids = test_inputs["decoder_input_ids"].squeeze()
            test_labels = test_inputs["labels"].squeeze()
            datable("test_input_ids", test_input_ids)
            datable("test_attention_mask", test_attention_mask)
            datable("test_decoder_input_ids", test_decoder_input_ids)
            datable("test_labels", test_labels)
            datable("test_inputs", test_inputs)

        return DataTableSet(datable)

    def _load_model_args(self, input_dir):
        args = Seq2SeqArgs()
        args.load(input_dir)
        return args

    def _get_inputs_dict(self, batch):
        # 源码  [512]
        # batch(dict)->1.source_ids(tensor) [16,512]->[batch_size,max_seq_length]
        #              2.source_mask(tensor) [16,512]->[batch_size,max_seq_length]
        #              3.target_ids(tensor) [16,512]->[batch_size,max_seq_length]
        # [1*512] [64*1*512]

        # device = self.device
        pad_token_id = self.encoder_tokenizer.pad_token_id
        source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]
        y_ids = y[:-1].contiguous()
        labels = y[1:].clone()
        labels[y[1:] == pad_token_id] = -100
        inputs = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "decoder_input_ids": y_ids,
            "labels": labels,
        }
        return inputs

    def load_and_cache_examples(self, data, evaluate=False, no_cache=False, verbose=True, silent=False):
        """
        Creates a T5Dataset from data.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        encoder_tokenizer = self.encoder_tokenizer
        decoder_tokenizer = self.decoder_tokenizer
        args = self.args

        # if not no_cache:
        #     no_cache = args.no_cache
        #
        # if not no_cache:
        #     os.makedirs(self.args.cache_dir, exist_ok=True)

        mode = "dev" if evaluate else "train"

        return SimpleSummarizationDataset(encoder_tokenizer, self.args, data, mode)


        # if args.dataset_class:
        #     CustomDataset = args.dataset_class
        #     return CustomDataset(encoder_tokenizer, decoder_tokenizer, args, data, mode)
        # else:
        #     if args.model_type in ["bart", "marian", "blender", "blender-large"]:
        #         return SimpleSummarizationDataset(encoder_tokenizer, self.args, data, mode)
        #     else:
        #         return Seq2SeqDataset(encoder_tokenizer, decoder_tokenizer, self.args, data, mode,)


if __name__ == "__main__":
    from cogagent.data.readers.KE_Blender_reader import KE_BlenderReader

    reader = KE_BlenderReader(raw_data_path="/home/nlp/CogAGENT/datapath/KE_Blender_data/")
    train_data, dev_data, test_data = reader.read_all()
    vocab = reader.read_vocab()

    processor = KE_BlenderProcessor(plm="bert-base-cased", vocab=vocab)
    dev_dataset = processor.process_dev(dev_data['dev_data'])
    test_dataset = processor.process_test(test_data['test_data'])
    train_dataset = processor.process_train(train_data['train_data'])
    print("end")
