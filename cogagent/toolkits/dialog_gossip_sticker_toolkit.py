import os
import torch
from cogagent.toolkits.base_toolkit import BaseToolkit
from cogagent.utils import load_model
from cogagent.models.chinese_gossip_dialog import ChineseGossipDialog
from cogagent.models.sticker_dialog import StickerDialogModel
from transformers import BertTokenizerFast, BertTokenizer
import torch
import json
from tqdm import tqdm
from PIL import Image, ImageSequence
import cv2
import numpy
import matplotlib.pyplot as plt


class DialogGossipStickerToolkit(BaseToolkit):
    """
    & 功能
    一人一句闲聊型对话,附加表情包
    [speaker1]是机器端
    [speaker2]是用户端
    & 参数
    model_path:保存的模型的路径
    dataset_name:训练的数据集名称
    model_name:模型的名字
    sticker_dataset_name:表情包数据集名字
    sticker_model_name:表情包模型名字
    sticker_model_path:表情包模型路径
    language:语言
    max_history_len：可见历史对话长度
    generate_max_len:生成文本的最大长度
    select_id:选择概率最大的第几个图片
    & 可以搭配的参数
    dataset_name = "ChineseGossipDialog50w"，model_name = "ChineseGossipDialog"
    """

    def __init__(self,
                 dataset_name=None,
                 model_name=None,
                 vocab_path=None,
                 model_path=None,
                 file_or_model=None,
                 sticker_dataset_name=None,
                 sticker_model_name=None,
                 sticker_model_path=None,
                 language=None,
                 max_history_len=3,
                 generate_max_len=32,
                 select_id=4,
                 id2img_path=None,
                 image_path=None) -> object:
        super().__init__()
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.vocab_path = vocab_path
        self.model_path = model_path
        self.file_or_model = file_or_model
        self.sticker_dataset_name = sticker_dataset_name
        self.sticker_model_name = sticker_model_name
        self.sticker_model_path = sticker_model_path
        self.language = language
        self.max_history_len = max_history_len
        self.generate_max_len = generate_max_len
        self.select_id = select_id
        self.id2img_path = id2img_path
        self.image_path = image_path

        self.max_token_len = 512

        if self.dataset_name == "ChineseGossipDialog" and self.model_name == "ChineseGossipDialog":
            self.tokenizer = BertTokenizerFast(
                vocab_file=vocab_path,
                cls_token="[CLS]",
                sep_token="[SEP]",
                pad_token="[PAD]")
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
        if self.sticker_dataset_name == "Mod" and self.sticker_model_name == "StickerDialog":
            self.sticker_tokenizer = BertTokenizer.from_pretrained('YeungNLP/clip-vit-bert-chinese-1M')
            special_tokens_dict = {
                'additional_special_tokens': ['[speaker1]', '[speaker2]']
            }
            self.sticker_tokenizer.add_special_tokens(special_tokens_dict)
            sticker_addition_dict = {}
            sticker_addition_dict["id2img"] = {}
            with open(self.id2img_path, encoding='utf-8') as f:
                id2img = json.load(f)
                self.id2imgpath = {}
                for id, img in tqdm(id2img.items()):
                    id = int(id)
                    self.id2imgpath[id] = img
                    sticker_addition_dict["id2imgpath"] = self.id2imgpath
            sticker_addition_dict["tokenizer"] = self.sticker_tokenizer
            self.sticker_model = StickerDialogModel(max_image_id=307,
                                                    pretrained_image_model_name_or_path='YeungNLP/clip-vit-bert-chinese-1M',
                                                    pretrained_model_name_or_path='bert-base-chinese',
                                                    pretrained_image_tokenizer_name_or_path='BertTokenizerFast',
                                                    addition=sticker_addition_dict,
                                                    image_path=self.image_path)
            self.sticker_model = load_model(self.sticker_model, self.sticker_model_path)

    def _process_ChineseGossipDialog_for_ChineseGossipDialog(self,
                                                             raw_dict,
                                                             dialogue_history,
                                                             dialogue_history_token=None):
        user_tokens = self.tokenizer.encode(raw_dict["sentence"], add_special_tokens=False)
        dialogue_history_token.append(user_tokens)
        input_ids = [self.tokenizer.cls_token_id]
        for history_item_tokens in dialogue_history_token[-self.max_history_len:]:
            input_ids.extend(history_item_tokens)
            input_ids.append(self.tokenizer.sep_token_id)
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        sticker_input_strings = ''
        for index, history_item_strings in enumerate(dialogue_history[-self.max_history_len:]):
            if index % 2 == 0:
                sticker_input_strings = sticker_input_strings + '[speaker1]' + history_item_strings
            if index % 2 == 1:
                sticker_input_strings = sticker_input_strings + '[speaker2]' + history_item_strings
        tokenized_result = self.tokenizer.encode_plus(text=sticker_input_strings,
                                                      padding="max_length",
                                                      add_special_tokens=True,
                                                      max_length=50,
                                                      truncation=True)
        sticker_input_ids = tokenized_result["input_ids"]
        sticker_attention_mask = tokenized_result["attention_mask"]
        if all(sticker_attention_mask) == 1:
            sticker_valid_token_len = 50
        else:
            sticker_valid_token_len = sticker_attention_mask.index(0)
        processed_dict = {"current_device_flag": torch.tensor(0),
                          "input_ids": input_ids}
        sticker_processed_dict = {"input_ids": torch.tensor(sticker_input_ids).unsqueeze(0),
                                  "valid_token_len": torch.tensor(sticker_valid_token_len).unsqueeze(0)}
        return (processed_dict, sticker_processed_dict,dialogue_history,dialogue_history_token)

    def _process_one(self, raw_dict,dialogue_history=None,dialogue_history_token=None):
        processed_dict = {}
        sticker_processed_dict = {}
        if self.dataset_name == "ChineseGossipDialog" and self.model_name == "ChineseGossipDialog":
            processed_dict, sticker_processed_dict,dialogue_history,dialogue_history_token = self._process_ChineseGossipDialog_for_ChineseGossipDialog(
                raw_dict,
                dialogue_history,
                dialogue_history_token)
        return (processed_dict, sticker_processed_dict,dialogue_history,dialogue_history_token)

    def infer_one(self, user_sentence=None, dialogue_history=None,dialogue_history_token=None):
        raw_dict = {}
        raw_dict["sentence"] = user_sentence
        raw_dict["character"] = "user"
        dialogue_history.append(raw_dict["sentence"])
        processed_dict, sticker_processed_dict,dialogue_history,dialogue_history_token = self._process_one(raw_dict,
                                                                                                           dialogue_history,
                                                                                                           dialogue_history_token)
        infer_sentence = self.model.predict(batch=processed_dict)[0]
        infer_img_id_list = self.sticker_model.predict(batch=sticker_processed_dict)[0]
        infer_img_id = infer_img_id_list[self.select_id]
        img_path = os.path.join(self.image_path, self.id2imgpath[infer_img_id.item()])
        # im = Image.open(img_path)
        # for frame in ImageSequence.Iterator(im):
        #     frame = frame.convert('RGB')
        #     cv2_frame = numpy.array(frame)
        #     plt.imshow(cv2_frame)
        #     plt.show()
        dialogue_history.append(infer_sentence)
        return infer_sentence,img_path,dialogue_history,dialogue_history_token


if __name__ == "__main__":
    from cogagent.utils.io_utils import load_file_path_yaml
    config = load_file_path_yaml("/data/hongbang/CogAGENT/demo/config.yaml")
    dialoggossiptoolkit = DialogGossipStickerToolkit(
        dataset_name="ChineseGossipDialog",
        model_name="ChineseGossipDialog",
        # vocab_path="/data/mentianyi/code/CogNLP/datapath/gossip_dialog/chinese_gossip_dialog/raw_data/vocab.txt",
        # model_path="/data/mentianyi/code/CogNLP/datapath/gossip_dialog/chinese_gossip_dialog/raw_data",
        file_or_model="file",
        sticker_dataset_name="Mod",
        sticker_model_name="StickerDialog",
        # sticker_model_path="/data/mentianyi/code/CogNLP/datapath/mm_dialog/mod/experimental_result/final--2023-01-17--14-00-56.81/model/checkpoint-780000/models.pt",
        language="chinese",
        max_history_len=3,
        generate_max_len=20,
        select_id=0,
        **config["sticker"],
        # id2img_path="/data/mentianyi/code/CogNLP/datapath/mm_dialog/mod/raw_data/id2img.json",
        # image_path="/data/mentianyi/code/CogNLP/datapath/mm_dialog/mod/raw_data/meme_set")
    )
    dialogue_history = []
    dialogue_history_token = []
    while True:
        try:
            user_sentence = input("user:")
            sentence,image,dialogue_history,dialogue_history_token = dialoggossiptoolkit.infer_one(user_sentence=user_sentence,
                                                                                                   dialogue_history=dialogue_history,
                                                                                                   dialogue_history_token=dialogue_history_token)
            print(sentence,image,dialogue_history,dialogue_history_token)

        except KeyboardInterrupt:
            print("end")
            break

    print("end")
