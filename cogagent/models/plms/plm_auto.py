import torch.nn as nn
from transformers import AutoModel


class PlmAutoModel(nn.Module):
    def __init__(self, pretrained_model_name):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self._plm = AutoModel.from_pretrained(pretrained_model_name)
        self.hidden_dim = self._plm.embeddings.position_embeddings.embedding_dim
        self.config = self._plm.config_class.from_pretrained(pretrained_model_name)

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"] if "attention_mask" in batch else None
        token_type_ids = batch["token_type_ids"] if "token_type_ids" in batch else None
        return self._plm(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids)
