from cogagent.data.datable import DataTable
from cogagent.data.datableset import DataTableSet
from transformers import BertTokenizer
from tqdm import tqdm
import transformers
from cogagent.data.processors.base_processor import BaseProcessor
from cogagent.data.readers.sclstm_reader import sclstm_multiwoz_reader

transformers.logging.set_verbosity_error()  # set transformers logging level


class Sclstm_Processor(BaseProcessor):
    def __init__(self, plm, max_token_len, vocab):
        super().__init__()
        # self.plm = plm
        # self.max_token_len = max_token_len
        self.vocab = vocab
        self.batch_size = 64
        #self.tokenizer = BertTokenizer.from_pretrained(plm)

    def _process(self, data):
        datable = DataTable()
        print("Processing data...")

        # for sentence, label in tqdm(zip(data['sentence'], data['label']), total=len(data['sentence'])):
        #     # token = self.tokenizer.encode(text=sentence, truncation=True, padding="max_length", add_special_tokens=True,
        #     #                               max_length=self.max_token_len)
        #     # datable("input_ids", token)
        #     tokenized_data = self.tokenizer.encode_plus(text=sentence,
        #                                                 padding="max_length",
        #                                                 add_special_tokens=True,
        #                                                 max_length=self.max_token_len)
        #     datable("input_ids", tokenized_data["input_ids"])
        #     datable("token_type_ids", tokenized_data["token_type_ids"])
        #     datable("attention_mask", tokenized_data["attention_mask"])
        #     datable("label", self.vocab["label_vocab"].label2id(label))

        percentage = 1.0
        dataset = sclstm_multiwoz_reader(raw_data_path="/home/nlp/CogAGENT/datapath/sclstm_multiwoz_data/resource")

        for i in range(dataset.n_batch['train']):
            input_var, label_var, feats_var, lengths, refs, featStrs, sv_indexes = dataset.next_batch()
            datable("input_var", input_var)
            datable("label_var", label_var)
            datable("feats_var", feats_var)
            datable("lengths", lengths)
            datable("refs", refs)
            datable("featStrs", featStrs)
            datable("sv_indexes", sv_indexes)
        return DataTableSet(datable)

    def process_train(self, dataset):
        datable = DataTable()
        print("Processing train data...")
        # dataset = sclstm_multiwoz_reader(raw_data_path="/home/nlp/CogAGENT/datapath/sclstm_multiwoz_data/resource")
        for i in range(dataset.train_data_length):
            input_var, label_var, feats_var, lengths, refs, featStrs, sv_indexes = dataset.next_batch()
            # input_var->tensor, label_var->tensor, feats_var->tensor, lengths->list->int, refs->tuple-str, featsStrs->tuple-str, sv_indexes->tuple-list-int,
            # input_var->[64,max_length,1392] labe_var->[64,max_length] feats_var->torch.Size([64, 571])

            # 传入所有类型数据
            input_var = input_var.squeeze()
            label_var = label_var.squeeze()
            feats_var = feats_var.squeeze()
            datable("train_input_var", input_var)
            datable("train_label_var", label_var)
            datable("train_feats_var", feats_var)
            datable("train_lengths", lengths)
            datable("train_refs", refs)
            datable("train_featStrs", featStrs)
            datable("train_sv_indexes", sv_indexes)
            # 非tensor类型数据
            datable.add_not2torch("train_lengths")
            datable.add_not2torch("train_refs")
            datable.add_not2torch("train_featStrs")
            datable.add_not2torch("train_sv_indexes")
        return DataTableSet(datable)

    def process_dev(self, dataset):
        datable = DataTable()
        print("Processing valid data...")
        # dataset = sclstm_multiwoz_reader(raw_data_path="/home/nlp/CogAGENT/datapath/sclstm_multiwoz_data/resource")
        for i in range(dataset.valid_data_length):
            dev_input_var, dev_label_var, dev_feats_var, dev_lengths, dev_refs, dev_featStrs, dev_sv_indexes = dataset.next_batch(data_type='valid')
            # input_var->tensor, label_var->tensor, feats_var->tensor, lengths->list->int, refs->tuple-str, featsStrs->tuple-str, sv_indexes->tuple-list-int,
            # input_var->[batch_size,max_length,vocb] labe_var->[batch_size,max_length] feats_var->torch.Size([64, 571])

            # tensor类型数据
            dev_input_var = dev_input_var.squeeze()
            dev_label_var = dev_label_var.squeeze()
            dev_feats_var = dev_feats_var.squeeze()
            datable("dev_input_var", dev_input_var)
            datable("dev_label_var", dev_label_var)
            datable("dev_feats_var", dev_feats_var)
            datable("dev_lengths", dev_lengths)
            datable("dev_refs", dev_refs)
            datable("dev_featStrs", dev_featStrs)
            datable("dev_sv_indexes", dev_sv_indexes)
            #  非tensor类型数据
            datable.add_not2torch("dev_lengths")
            datable.add_not2torch("dev_refs")
            datable.add_not2torch("dev_featStrs")
            datable.add_not2torch("dev_sv_indexes")

        return DataTableSet(datable)


    def process_test(self, dataset):
        datable = DataTable()
        print("Processing test data...")
       # dataset = sclstm_multiwoz_reader(raw_data_path="/home/nlp/CogAGENT/datapath/sclstm_multiwoz_data/resource")
        n_batch = 64
        for i in range(dataset.test_data_length):
            test_input_var, test_label_var, test_feats_var, test_lengths, test_refs, test_featStrs, test_sv_indexes = dataset.next_batch(data_type='test')
            # input_var->tensor, label_var->tensor, feats_var->tensor, lengths->list->int, refs->tuple-str, featsStrs->tuple-str, sv_indexes->tuple-list-int,
            # tensor类型数据
            test_input_var = test_input_var.squeeze()
            test_label_var = test_label_var.squeeze()
            test_feats_var = test_feats_var.squeeze()
            datable("test_input_var", test_input_var)
            datable("test_label_var", test_label_var)
            datable("test_feats_var", test_feats_var)
            datable("test_lengths", test_lengths)
            datable("test_refs", test_refs)
            datable("test_featStrs", test_featStrs)
            datable("test_sv_indexes", test_sv_indexes)
            # 非tensor类型数据
            datable.add_not2torch("test_lengths")
            datable.add_not2torch("test_refs")
            datable.add_not2torch("test_featStrs")
            datable.add_not2torch("test_sv_indexes")
        return DataTableSet(datable)


if __name__ == "__main__":
    from cogagent.data.readers.sclstm_reader import sclstm_multiwoz_reader
    from cogagent.data.readers.sst2_reader import Sst2Reader


    dataset = sclstm_multiwoz_reader(raw_data_path="/home/nlp/CogAGENT/datapath/sclstm_multiwoz_data/resource")
    train_data, dev_data, test_data = dataset.read_all()
    vocab = dataset.read_vocab(vocab_file_path="/home/nlp/CogAGENT/datapath/sclstm_multiwoz_data/resource/vocab.txt")

    # processor = Sst2Processor(plm="bert-base-cased", max_token_len=128, vocab=vocab)
    processor = Sclstm_Processor(plm="bert-base-cased", max_token_len=128, vocab=vocab)

    train_dataset = processor.process_train(dataset)
    dev_dataset = processor.process_dev(dataset)
    test_dataset = processor.process_test(dataset)
    look1 = train_dataset.datable.datas['train_input_var']

    print("end")
