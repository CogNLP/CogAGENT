from cogagent.models.base_model import BaseModel
import copy
import torch
import os
import imghdr
from io import BytesIO
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel,BertForSequenceClassification
import torchvision.transforms as transforms

class BertModel(BertForSequenceClassification):
    def __init__(self, config):
        config.num_labels = 2
        super().__init__(config)



class StickerDialogModel(BaseModel):
    """
    & 功能
    图像分类:输入一个三位的RGB图像，大小为256*256，对图像进行分类
    & 参数
    vocab,词表
    baseline,有"cnn"和“lstm”和"mlp"和"transformer"
    & 模型

    """

    def __init__(self,
                 max_image_id,
                 pretrained_model_name_or_path,
                 pretrained_image_tokenizer_name_or_path,
                 pretrained_image_model_name_or_path,
                 addition,
                 image_path
                 ):
        super().__init__()
        self.max_image_id = max_image_id
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.pretrained_image_tokenizer_name_or_path = pretrained_image_tokenizer_name_or_path
        self.pretrained_image_model_name_or_path=pretrained_image_model_name_or_path
        self.addition = addition
        self.image_path = image_path
        self.id2img = addition["id2img"]
        self.id2imgpath = addition["id2imgpath"]
        self.text_tokenizer = addition["tokenizer"]

        CLIPProcessor.tokenizer_class = pretrained_image_tokenizer_name_or_path
        self.img_tokenizer = CLIPProcessor.from_pretrained(pretrained_image_model_name_or_path)
        self.img_model = CLIPVisionModel.from_pretrained(pretrained_image_model_name_or_path)
        self.bert = BertModel.from_pretrained( pretrained_model_name_or_path)
        self.bert.resize_token_embeddings(len(self.text_tokenizer))
        self.img_ff = torch.nn.Linear(768, 768)

        self.transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275,  0.40821073],
                                 [0.26862954, 0.26130258, 0.27577711])
        ])

        self.img_model.eval()
        for p in self.img_model.parameters():
            p.requires_grad = False


        self.prepare_valid()

    # def loss(self, batch, loss_function):
    #     cls_loss = self.forward(batch)
    #     return cls_loss

    def loss(self, batch, loss_function):
        batch_device = batch['input_ids'].device
        batch_size = batch['input_ids'].shape[0]
        res, neg_res = self.forward(batch)
        labels = torch.ones(batch_size, dtype=torch.long, device=batch_device)
        neg_labels = torch.zeros(batch_size, dtype=torch.long, device=batch_device)
        loss = loss_function(res, labels)
        neg_loss = loss_function(neg_res, neg_labels)
        cls_loss = (loss + neg_loss) / 2
        final_loss = cls_loss
        return final_loss

    def forward(self, batch):
        batch_device = batch['input_ids'].device
        batch_size = batch['input_ids'].shape[0]
        # batch*512(embedding_dim)的图片表示
        img_emb = self.get_emb_by_imgids(batch["img_id"], batch_device=batch_device)
        neg_img_emb = self.get_emb_by_imgids(batch["neg_img_id"], batch_device=batch_device)

        # batch*1*768(embedding_dim)的[SEP]表示
        sep_emb = self.bert.bert.embeddings.word_embeddings(
            torch.tensor(self.text_tokenizer.sep_token_id, dtype=torch.long, device=batch_device)).unsqueeze(0).unsqueeze(
            0).repeat(batch_size, 1,
                      1)
        # batch*max_len*768(embedding_dim)
        text_emb = self.bert.bert.embeddings.word_embeddings(batch['input_ids'])
        # batch*1*768(embedding_dim)
        img_emb = self.img_ff(img_emb).unsqueeze(1)
        # batch*1*768(embedding_dim)
        neg_img_emb =self.img_ff(neg_img_emb).unsqueeze(1)
        # batch*(文本长度+1条图片表示+1条分隔符表示)*768
        input_emb = torch.cat([text_emb, img_emb, sep_emb], dim=1)
        neg_input_emb = torch.cat([text_emb, neg_img_emb, sep_emb], dim=1)
        # batch*2
        ones_mask2 = torch.ones(batch_size, 2, device=batch_device)
        # batch*(文本长度+1条图片表示+1条分隔符表示)*768
        noaddocr_attention_mask = torch.cat([batch["attention_mask"], ones_mask2], dim=1)
        # batch*(文本长度+1条图片表示+1条分隔符表示)*768
        token_type_ids = torch.zeros_like(noaddocr_attention_mask, dtype=torch.long)
        # batch*(文本长度+1条图片表示+1条分隔符表示)*768 最后两列变成1
        token_type_ids[:, -2:] = 1

        labels = torch.ones(batch_size, dtype=torch.long, device=batch_device)
        neg_labels = torch.zeros(batch_size, dtype=torch.long, device=batch_device)

        res = self.bert(inputs_embeds=input_emb,
                        attention_mask=noaddocr_attention_mask,
                        token_type_ids=token_type_ids,
                        return_dict=True,
                        labels=labels,
                        output_hidden_states=True)
        neg_res = self.bert(inputs_embeds=neg_input_emb,
                            attention_mask=noaddocr_attention_mask,
                            token_type_ids=token_type_ids,
                            return_dict=True,
                            labels=neg_labels,
                            output_hidden_states=True)
        res = res.logits
        neg_res = neg_res.logits
        return (res,neg_res)

    def evaluate(self, batch, metric_function):
        pred = self.predict(batch)
        metric_function.evaluate(pred, batch["img_id"])

    def predict(self, batch):
        batch_device = batch['input_ids'].device
        batch_size = batch['input_ids'].shape[0]
        all_img_embs = self.all_img_embs
        all_img_embs = all_img_embs.unsqueeze(1).to(batch_device)
        img_num = all_img_embs.size(0)
        pred = []

        for i in range(batch_size):
            valid_token_len = batch["valid_token_len"][i]
            input_ids = batch["input_ids"][i][:valid_token_len]
            img_emb= copy.deepcopy(all_img_embs)
            img_emb = self.img_ff(img_emb)
            sep_emb = self.bert.bert.embeddings.word_embeddings(
                torch.tensor(self.text_tokenizer.sep_token_id, device=batch_device, dtype=torch.long)).unsqueeze(
                0).unsqueeze(0).repeat(img_num, 1, 1)
            text_emb = self.bert.bert.embeddings.word_embeddings(input_ids)
            text_emb = text_emb.repeat(img_num, 1, 1)
            input_emb = torch.cat([text_emb, img_emb, sep_emb], dim=1)
            extra_len = input_emb.size(1) - text_emb.size(1)
            token_type_ids = torch.zeros(size=(input_emb.size(0), input_emb.size(1)), device=batch_device,
                                         dtype=torch.long)
            token_type_ids[:, -extra_len:] = 1
            res = self.bert(inputs_embeds=input_emb,
                            token_type_ids=token_type_ids,
                            return_dict=True)
            logits = res.logits[:, 1]
            _, idx = torch.sort(logits, descending=True)
            idx=idx.cpu()
            return_pred = torch.arange(idx.size(0))[idx].tolist()
            pred.append(idx)
        return pred

    def prepare_valid(self):
        with torch.no_grad():
            img_objs = []
            for id in range(self.max_image_id):
                img_obj = self.get_image_obj(id)
                img_objs.append(img_obj)
            img_tokens = self.img_tokenizer(images=img_objs, return_tensors="pt")
            self.img_model.eval()
            self.all_img_embs = self.img_model(**img_tokens).pooler_output

    def get_emb_by_imgids(self, img_ids, batch_device):
        # 输入：一个list的图片id
        # 输出：图片的表示
        img_objs = []
        for i, id in enumerate(img_ids):
            # c*h*w 图片的三通道，这里clip是3*224*224
            img_obj = self.get_image_obj(id)
            img_objs.append(img_obj)
        # ?这里为什么要对图片做tokenize呢
        img_tokens = self.img_tokenizer(images=img_objs, return_tensors="pt")
        img_tokens.data["pixel_values"]=img_tokens.data["pixel_values"].to(batch_device)
        # 返回b*embedding_dim的向量表示
        img_emb = self.img_model(**img_tokens).pooler_output
        return img_emb

    def get_image_obj(self, id):
        # 输入：一个图片id
        # 输出：一个图片的三维矩阵

        # 图片在预存数据集中的情况
        if id in self.id2img:
            return self.id2img[id]

        # 图片不在预存数据集中的情况
        img_file_name = self.id2imgpath[id]
        path = os.path.join(self.image_path, img_file_name)
        with open(path, 'rb') as fin:
            data = fin.read()
        # 读取静态图或者gif的某一个时刻,能把不同大小的图片都处理成3*224*224

        img_obj = self.transform(self.pick_single_frame(data))
        # id2img在id到三维图像的映射添加内容
        self.id2img[id] = img_obj
        return img_obj

    def pick_single_frame(self, data: bytes) -> Image.Image:
        # 解析图片二进制数据到PIL Image对象，如果是静态图片则直接读取，如果是gif则解析第一帧
        if self.judge_img_type(data) != 'gif':
            img = self.read_pil_image(data)
        else:
            img = self.get_gif_nframe(data, 0)  # 取gif的第0帧
        img = self.remove_transparency(img)
        img = img.convert('RGB')
        return img

    def judge_img_type(self, data):
        # 从内存数据判断图像格式
        # all types: https://docs.python.org/zh-cn/3/library/imghdr.html
        img_type = imghdr.what(BytesIO(data))
        return img_type

    def read_pil_image(self, data):
        # 从内存data读取PIL格式
        img = Image.open(BytesIO(data))
        return img

    def get_gif_nframe(self, data, n):
        # 读取gif的第n帧, 从0开始计数
        # n支持负数(表示倒数第|n|张)
        if self.judge_img_type(data) != 'gif':
            return None
        gif = Image.open(BytesIO(data))
        if not -gif.n_frames <= n <= gif.n_frames - 1:
            return None
        if n < 0:
            n = gif.n_frames + n
        gif.seek(n)
        img = gif.convert('RGBA')
        return img

    def remove_transparency(self, img_pil, bg_colour=(255, 255, 255)):
        # 删除alpha通道
        # Only process if image has transparency
        if img_pil.mode in ('RGBA', 'LA') or \
                (img_pil.mode == 'P' and 'transparency' in img_pil.info):
            # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
            alpha = img_pil.convert('RGBA').getchannel('A')

            # Create a new background image of our matt color.
            # Must be RGBA because paste requires both images have the same format
            # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
            new_img = Image.new("RGBA", img_pil.size, bg_colour + (255,))
            new_img.paste(img_pil, mask=alpha)
            return new_img
        else:
            return img_pil

