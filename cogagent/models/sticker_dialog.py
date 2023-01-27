from cogagent.models.base_model import BaseModel
import copy
import torch
import os
import imghdr
from io import BytesIO
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel, BertForSequenceClassification
import torchvision.transforms as transforms


class BertModel(BertForSequenceClassification):
    def __init__(self, config):
        config.num_labels = 2
        super().__init__(config)


class StickerDialogModel(BaseModel):

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
        self.pretrained_image_model_name_or_path = pretrained_image_model_name_or_path
        self.addition = addition
        self.image_path = image_path
        self.id2img = addition["id2img"]
        self.id2imgpath = addition["id2imgpath"]
        self.text_tokenizer = addition["tokenizer"]

        CLIPProcessor.tokenizer_class = pretrained_image_tokenizer_name_or_path
        self.img_tokenizer = CLIPProcessor.from_pretrained(pretrained_image_model_name_or_path)
        self.img_model = CLIPVisionModel.from_pretrained(pretrained_image_model_name_or_path)
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.bert.resize_token_embeddings(len(self.text_tokenizer))
        self.img_ff = torch.nn.Linear(768, 768)

        self.transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                 [0.26862954, 0.26130258, 0.27577711])
        ])

        self.img_model.eval()
        for p in self.img_model.parameters():
            p.requires_grad = False

        self.prepare_valid()

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
        img_emb = self.get_emb_by_imgids(batch["img_id"], batch_device=batch_device)
        neg_img_emb = self.get_emb_by_imgids(batch["neg_img_id"], batch_device=batch_device)

        sep_emb = self.bert.bert.embeddings.word_embeddings(
            torch.tensor(self.text_tokenizer.sep_token_id, dtype=torch.long, device=batch_device)).unsqueeze(
            0).unsqueeze(
            0).repeat(batch_size, 1,
                      1)

        text_emb = self.bert.bert.embeddings.word_embeddings(batch['input_ids'])

        img_emb = self.img_ff(img_emb).unsqueeze(1)

        neg_img_emb = self.img_ff(neg_img_emb).unsqueeze(1)

        input_emb = torch.cat([text_emb, img_emb, sep_emb], dim=1)
        neg_input_emb = torch.cat([text_emb, neg_img_emb, sep_emb], dim=1)

        ones_mask2 = torch.ones(batch_size, 2, device=batch_device)

        noaddocr_attention_mask = torch.cat([batch["attention_mask"], ones_mask2], dim=1)

        token_type_ids = torch.zeros_like(noaddocr_attention_mask, dtype=torch.long)

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
        return (res, neg_res)

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
            img_emb = copy.deepcopy(all_img_embs)
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
            idx = idx.cpu()
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

        img_objs = []
        for i, id in enumerate(img_ids):
            img_obj = self.get_image_obj(id)
            img_objs.append(img_obj)

        img_tokens = self.img_tokenizer(images=img_objs, return_tensors="pt")
        img_tokens.data["pixel_values"] = img_tokens.data["pixel_values"].to(batch_device)

        img_emb = self.img_model(**img_tokens).pooler_output
        return img_emb

    def get_image_obj(self, id):

        if id in self.id2img:
            return self.id2img[id]

        img_file_name = self.id2imgpath[id]
        path = os.path.join(self.image_path, img_file_name)
        with open(path, 'rb') as fin:
            data = fin.read()

        img_obj = self.transform(self.pick_single_frame(data))

        self.id2img[id] = img_obj
        return img_obj

    def pick_single_frame(self, data: bytes) -> Image.Image:

        if self.judge_img_type(data) != 'gif':
            img = self.read_pil_image(data)
        else:
            img = self.get_gif_nframe(data, 0)
        img = self.remove_transparency(img)
        img = img.convert('RGB')
        return img

    def judge_img_type(self, data):

        img_type = imghdr.what(BytesIO(data))
        return img_type

    def read_pil_image(self, data):

        img = Image.open(BytesIO(data))
        return img

    def get_gif_nframe(self, data, n):

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

        if img_pil.mode in ('RGBA', 'LA') or \
                (img_pil.mode == 'P' and 'transparency' in img_pil.info):

            alpha = img_pil.convert('RGBA').getchannel('A')

            new_img = Image.new("RGBA", img_pil.size, bg_colour + (255,))
            new_img.paste(img_pil, mask=alpha)
            return new_img
        else:
            return img_pil
