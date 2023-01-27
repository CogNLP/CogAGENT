import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
import torch.nn.functional as F
import torch


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]

        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)

        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p

        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


class ChineseGossipDialog(nn.Module):

    def __init__(self,
                 generate_max_len,
                 addition,
                 valid_mod,
                 pretrained_model_name_or_path,
                 file_or_model="file"):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.generate_max_len = generate_max_len
        self.addition = addition
        self.valid_mod = valid_mod
        self.file_or_model = file_or_model
        if file_or_model == "file":
            self.plm = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path)
        else:
            self.plm = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

        self.repetition_penalty = 1.2
        self.topk = 5
        self.topp = 0.95

    def loss(self, batch, loss_function):

        pred = self.forward(batch)

        labels = batch["input_ids"]

        shift_logits = pred[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        # view将不同batch拼接成一个长的文本序列
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        loss = loss_function(shift_logits, shift_labels)
        return loss

    def forward(self, batch):

        x = self.plm(input_ids=batch["input_ids"],
                     attention_mask=batch["attention_mask"])[0]
        return x

    def evaluate(self, batch, metric_function):
        if self.valid_mod == "parallel_valid":
            pred = self.predict(batch)
            label = batch["input_ids"]
            label = label[:, 1:].contiguous()
            label = label.view(-1)

            non_pad_mask = label.ne(self.addition["pad_id"])
            pred = pred.masked_select(non_pad_mask)
            label = label.masked_select(non_pad_mask)
            pred = pred.tolist()
            label = label.tolist()
            metric_function.evaluate(pred, label)
        if self.valid_mod == "serial_valid":
            pred = self.predict(batch)
            label = batch["labels"]
            metric_function.evaluate(pred, label)

    def predict(self, batch):
        if self.valid_mod == "parallel_valid":
            pred = self.forward(batch)
            shift_logits = pred[:, :-1, :].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            _, shift_logits = shift_logits.max(dim=-1)
            return shift_logits
        if self.valid_mod == "serial_valid":
            batch_device = batch["current_device_flag"].device

            pred = []

            for article_id in batch['input_ids']:
                if str(batch_device) != "cpu":
                    curr_input_tensor = torch.tensor(article_id).to(batch_device)
                else:
                    curr_input_tensor = article_id

                generated = []

                for _ in range(self.generate_max_len):

                    last_hidden_state = self.plm(input_ids=curr_input_tensor)[0]

                    next_token_logits = last_hidden_state[-1, :]

                    for existed_id in set(generated):
                        next_token_logits[existed_id] = next_token_logits[existed_id] / self.repetition_penalty

                    next_token_logits[self.addition["tokenizer"].convert_tokens_to_ids('[UNK]')] = -float('Inf')

                    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=self.topk, top_p=self.topp)

                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

                    if next_token.item() == self.addition["tokenizer"].sep_token_id or len(curr_input_tensor) > 511:
                        break
                    generated.append(next_token.item())

                    curr_input_tensor = torch.cat((curr_input_tensor, next_token), dim=0)
                text = self.addition["tokenizer"].convert_ids_to_tokens(generated)
                text = "".join(text)
                pred.append(text)
            return pred
