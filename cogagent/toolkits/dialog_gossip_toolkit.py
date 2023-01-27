from cogagent.toolkits.base_toolkit import BaseToolkit
from cogagent.utils import load_model
from cogagent.models.chinese_gossip_dialog import ChineseGossipDialog
from transformers import BertTokenizer, BertTokenizerFast
import torch


class DialogGossipToolkit(BaseToolkit):

    def __init__(self,
                 vocab_path=None,
                 file_or_model=None,
                 model_path=None,
                 dataset_name=None,
                 model_name=None,
                 language=None,
                 max_history_len=3,
                 generate_max_len=32):
        super().__init__()
        self.vocab_path = vocab_path
        self.file_or_model = file_or_model
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.language = language
        self.max_token_len = max_history_len
        self.generate_max_len = generate_max_len

        self.max_token_len = 512
        self.history_tokens = []

        if self.dataset_name == "ChineseGossipDialog" and self.model_name == "ChineseGossipDialog":
            if vocab_path is not None:
                self.tokenizer = BertTokenizerFast(vocab_file=vocab_path,
                                                   sep_token="[SEP]",
                                                   pad_token="[PAD]",
                                                   cls_token="[CLS]")
            else:
                self.tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
            self.cls_id = self.tokenizer.cls_token_id
            self.sep_id = self.tokenizer.sep_token_id
            self.pad_id = self.tokenizer.pad_token_id
            addition_dict = {}
            addition_dict["cls_id"] = self.cls_id
            addition_dict["sep_id"] = self.sep_id
            addition_dict["pad_id"] = self.pad_id
            addition_dict["valid_mod"] = "serial_valid"
            addition_dict["tokenizer"] = self.tokenizer
            self.model = ChineseGossipDialog(
                pretrained_model_name_or_path=self.model_path,
                file_or_model=file_or_model,
                generate_max_len=20,
                addition=addition_dict,
                valid_mod="serial_valid")
            if file_or_model == "model":
                self.model = load_model(self.model, self.model_path)

    def _process_ChineseGossipDialog50w_for_ChineseGossipDialog(self, raw_dict):
        user_tokens = self.tokenizer.encode(raw_dict["sentence"], add_special_tokens=False)
        self.history_tokens.append(user_tokens)
        input_ids = [self.tokenizer.cls_token_id]
        for history_item_tokens in self.history_tokens[-self.max_token_len:]:
            input_ids.extend(history_item_tokens)
            input_ids.append(self.tokenizer.sep_token_id)
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        processed_dict = {"current_device_flag": torch.tensor(0),
                          "input_ids": input_ids}
        return processed_dict

    def _process_one(self, raw_dict):
        processed_dict = {}
        if self.dataset_name == "ChineseGossipDialog" and self.model_name == "ChineseGossipDialog":
            processed_dict = self._process_ChineseGossipDialog50w_for_ChineseGossipDialog(raw_dict)
        return processed_dict

    def infer_one(self, raw_dict=None):
        dialog_history_dict = {"dialog_history": []}
        while True:
            try:
                user_sentence = input("user:")
                raw_dict = {}
                raw_dict["sentence"] = user_sentence
                raw_dict["character"] = "user"
                processed_dict = self._process_one(raw_dict)
                infer_sentence = self.model.predict(batch=processed_dict)[0]
                print(infer_sentence)
                infer_dict = {}
                infer_dict["sentence"] = infer_sentence
                infer_dict["character"] = "agent"
                dialog_history_dict["dialog_history"].append(raw_dict)
                dialog_history_dict["dialog_history"].append(infer_dict)
            except KeyboardInterrupt:
                print("聊天结束.")
                break
        return dialog_history_dict


if __name__ == "__main__":
    dialoggossiptoolkit = DialogGossipToolkit(
        dataset_name="ChineseGossipDialog",
        model_name="ChineseGossipDialog",
        # 以下三行是别人训练的文件
        vocab_path="/data/mentianyi/code/CogNLP/datapath/gossip_dialog/chinese_gossip_dialog/raw_data/vocab.txt",
        model_path="/data/mentianyi/code/CogNLP/datapath/gossip_dialog/chinese_gossip_dialog/raw_data",
        file_or_model="file",
        # 以下两行是自己训练的文件
        # model_path="/data/mentianyi/code/CogNLP/datapath/gossip_dialog/chinese_gossip_dialog/experimental_result/final--2023-01-13--04-24-43.18/model/checkpoint-40000/models.pt",
        # file_or_model="model",
        language="chinese",
        max_history_len=3,
        generate_max_len=20)
    infer_dict = dialoggossiptoolkit.infer_one()
    print(infer_dict)
    print("end")
