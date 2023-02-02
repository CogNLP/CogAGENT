from cogagent.models.base_model import BaseModel
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
import torch


class BaseReadingComprehensionModel(BaseModel):
    def __init__(self, vocab, plm):
        super().__init__()
        self.vocab = vocab
        self.plm = plm

        self.bert = BertModel.from_pretrained(plm)
        self.input_size = 768
        self.classes_num = 2
        self.linear = nn.Linear(in_features=self.input_size, out_features=self.classes_num)

    def loss(self, batch_dict, loss_function):
        input_ids, attention_mask, token_type_ids, \
        start_positions, end_positions = self.get_batch(batch_dict)

        start_logits,end_logits = self.forward(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )

        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        start_loss = loss_function(start_logits, start_positions)
        end_loss = loss_function(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        return total_loss

    def forward(self, input_ids,attention_mask,token_type_ids,):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        logits = self.linear(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        return start_logits,end_logits

    def evaluate(self, batch, metric_function):
        input_ids, attention_mask, token_type_ids, \
        start_positions, end_positions = self.get_batch(batch)
        # print(start_positions,end_positions)

        start_logits, end_logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        metric_function.evaluate(
            start_logits,end_logits,batch
        )
        # start_positions_pred = torch.argmax(start_logits,dim=-1)
        # end_positions_pred = torch.argmax(end_logits,dim=-1)
        #
        # metric_function.evaluate(start_positions_pred,end_positions_pred,start_positions,end_positions)


    def predict(self, batch):
        input_ids, attention_mask, token_type_ids, \
        start_positions, end_positions = self.get_batch(batch)

        start_logits, end_logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return start_logits,end_logits,batch

    def get_batch(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        start_positions = batch["start_position"]
        end_positions = batch["end_position"]
        return input_ids,attention_mask,token_type_ids,start_positions,end_positions
