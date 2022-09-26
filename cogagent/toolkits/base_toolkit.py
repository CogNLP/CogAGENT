from transformers import BertTokenizer
from cogagent.utils.vocab_utils import Vocabulary
import torch

class BaseToolkit:
    def __init__(self,bert_model=None,model_path=None,vocabulary_path=None,device=None):
        self.bert_model = bert_model
        self.model_path = model_path
        self.vocabulary_path = vocabulary_path
        self.device = device
        if self.bert_model:
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)
        if self.vocabulary_path:
            self.vocabulary = Vocabulary()
            self.vocabulary.load(vocabulary_path)




