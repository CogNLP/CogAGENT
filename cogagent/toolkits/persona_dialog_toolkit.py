from cognlp.toolkits.base_toolkit import BaseToolkit
from cognlp.utils import load_model
from cognlp.models.bob import Bob
import torch
from transformers import BertTokenizer
import copy
import numpy as np


class PersonaDialogToolkit(BaseToolkit):

    def __init__(self,
                 model_path=None,
                 dataset_name=None,
                 model_name=None,
                 language=None,
                 device="cpu"):
        super().__init__()
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.language = language
        self.device = device

        if self.dataset_name == "Convai2" and self.model_name == "Bob":
            self.max_source_len = 64
            self.max_target_len = 32
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            self.model = Bob(
                encoder_model='bert-base-uncased',
                decoder_model='bert-base-uncased',
                decoder_model2='bert-base-uncased',
                nli_batch_size=32,
                addition=None)
            self.model = load_model(self.model, self.model_path)
            self.model = self.model.to(self.device)

    def _get_tokenized_data(self, data, max_len):
        tokenized_data = self.tokenizer(data,
                                        truncation=True,
                                        padding=True,
                                        max_length=max_len)

        input_ids_data = tokenized_data["input_ids"]
        token_type_ids_data = tokenized_data['token_type_ids']
        attention_mask_data = tokenized_data['attention_mask']

        input_ids_data += [0 for _ in range(max_len - len(input_ids_data))]
        token_type_ids_data += [0 for _ in range(max_len - len(token_type_ids_data))]
        attention_mask_data += [0 for _ in range(max_len - len(attention_mask_data))]

        return [input_ids_data, token_type_ids_data, attention_mask_data]

    def _process_Convai2_for_Bob(self, raw_dict):
        input_ids_persona, token_type_ids_persona, \
        attention_mask_persona = self._get_tokenized_data(data=raw_dict['persona_sentence'],
                                                          max_len=self.max_source_len)

        input_ids_query, token_type_ids_query, \
        attention_mask_query = self._get_tokenized_data(data=raw_dict['query_sentence'], max_len=self.max_source_len)

        # fake
        input_ids_response, token_type_ids_response, \
        attention_mask_response = self._get_tokenized_data(data=raw_dict['persona_sentence'],
                                                           max_len=self.max_target_len)

        input_ids = copy.deepcopy(input_ids_persona)
        attention_mask = copy.deepcopy(attention_mask_persona)
        token_type_ids = copy.deepcopy(token_type_ids_persona)
        token_type_ids = token_type_ids * 1
        input_ids.extend(input_ids_query)
        attention_mask.extend(attention_mask_query)
        token_type_ids.extend(token_type_ids_query)

        mask_flag = torch.BoolTensor(1 - np.array(attention_mask_response))
        input_ids_response = torch.tensor(input_ids_response)
        lables = input_ids_response.masked_fill(mask_flag, -100)

        processed_dict = {"current_device_flag": torch.tensor(0).to(self.device),
                          "input_ids_persona": torch.tensor(input_ids_persona).unsqueeze(0).to(self.device),
                          "input_ids_query": torch.tensor(input_ids_query).unsqueeze(0).to(self.device),
                          "input_ids": torch.tensor(input_ids).unsqueeze(0).to(self.device),
                          "token_type_ids": torch.tensor(token_type_ids).unsqueeze(0).to(self.device),
                          "attention_mask": torch.tensor(attention_mask).unsqueeze(0).to(self.device),
                          "decoder_input_ids_response": torch.tensor(input_ids_persona).unsqueeze(0).to(self.device),
                          "decoder_token_type_ids_response": torch.tensor(token_type_ids_response).unsqueeze(0).to(
                              self.device),
                          "decoder_attention_mask_response": torch.tensor(attention_mask_response).unsqueeze(0).to(
                              self.device),
                          "lables": lables.unsqueeze(0).to(self.device),
                          }
        return processed_dict

    def _process_one(self, raw_dict):
        processed_dict = {}
        if self.dataset_name == "Convai2" and self.model_name == "Bob":
            processed_dict = self._process_Convai2_for_Bob(raw_dict)
        return processed_dict

    def infer_one(self, raw_dict=None):
        dialog_history_dict = {"dialog_history": []}
        persona_sentence = input("persona:")
        while True:
            try:
                query_sentence = input("query:")
                raw_dict = {}
                raw_dict["persona_sentence"] = persona_sentence
                raw_dict["query_sentence"] = query_sentence
                processed_dict = self._process_one(raw_dict)
                infer_sentence = self.model.predict(batch=processed_dict)["example"][0]
                del infer_sentence['gold_token']
                print(infer_sentence)
                dialog_history_dict["dialog_history"].append(infer_sentence)
            except KeyboardInterrupt:
                print("聊天结束.")
                break
        return dialog_history_dict


if __name__ == "__main__":
    personadialogtoolkit = PersonaDialogToolkit(
        dataset_name="Convai2",
        model_name="Bob",
        model_path="/data/mentianyi/code/CogNLP/datapath/controllable_dialog/convai2/experimental_result/final--2023-01-17--14-25-23.70/model/checkpoint-139200/models.pt",
        language="english",
        device="cuda:0")
    infer_dict = personadialogtoolkit.infer_one()
    print(infer_dict)
    print("end")
    # final--2023-01-17--14-25-23.70/model/checkpoint-139200
    # final--2023-01-17--03-08-21.39/model/checkpoint-153600

    # persona:  i read twenty books a year. i am a stunt double as my second job. i only eat kosher.
    # query: hello what are doing today ?
    # gold: i am good , i just got off work and tired , i have two jobs .
    # response from D1:
    # response from D2:
