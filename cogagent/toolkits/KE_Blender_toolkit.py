from cogagent.toolkits.base_toolkit import BaseToolkit
from cogagent.utils.train_utils import move_dict_value_to_device
from cogagent.models.KE_Blender.KE_Blender_model import BlenderbotSmallForConditionalGeneration
from transformers import (
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
)

import torch

class KE_BlenderToolkit(BaseToolkit):

    def __init__(self, bert_model=None, model_path=None, vocabulary_path=None, device=None):
        super().__init__()
        self.model = BlenderbotSmallForConditionalGeneration.from_pretrained(model_path)
        self.encoder_tokenizer = BlenderbotSmallTokenizer.from_pretrained(model_path, additional_special_tokens=['__defi__', '__hype__', '[MASK]'])
        self.decoder_tokenizer = self.encoder_tokenizer
        self.model.resize_token_embeddings(len(self.encoder_tokenizer))
        self.device = device
        self.model.to(self.device)

    def generate(self, sentence):
        all_outputs = []
        for batch in [sentence[i:] for i in range(0, len(sentence), )]:
            input_ids = self.encoder_tokenizer.batch_encode_plus(
                batch, max_length=512, padding='max_length', truncation=True, return_tensors="pt",
            )["input_ids"]
            input_ids = input_ids.to(self.device)
            outputs = self.model.generate(
                input_ids=input_ids,
                num_beams=1,
                max_length=128,
                length_penalty=2.0,
                early_stopping=True,
                repetition_penalty=1.0,
                do_sample=False,
                top_k=10,
                top_p=1.0,
                num_return_sequences=1,
                # temperature=0.7
            )
            all_outputs.extend(outputs.cpu().numpy())
        outputs = [
            self.decoder_tokenizer.decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for output_id in all_outputs
        ]
        return outputs

if __name__ == '__main__':
    from cogagent.toolkits.KE_Blender_toolkit import KE_BlenderToolkit
    toolkit = KE_BlenderToolkit(model_path='/home/nlp/CogAGENT/datapath/KE_Blender_data', device=torch.device("cuda:1"))
    sentence = ["Who are you?"]
    label = toolkit.generate(sentence)
    print("label:", label)


