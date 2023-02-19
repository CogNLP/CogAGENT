from cogagent.toolkits.base_toolkit import BaseToolkit
from cogagent.data.processors.sst2_processors.sst2_processor import text_classification_for_sst2
from cogagent.models.base_text_classification_model import BaseTextClassificationModel
from cogagent.models.plms.plm_auto import PlmAutoModel
from collections import defaultdict
from cogagent import Vocabulary,load_model
from cogagent.utils.train_utils import move_dict_value_to_device
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

class DialogeSafetyDetectionToolkit(BaseToolkit):
    def __init__(self,plm_name,model_path,device):
        super(DialogeSafetyDetectionToolkit, self).__init__()

        label_vocab = Vocabulary()
        label_vocab.add_dict(
            {"Safe": 0,
             "Unsafe": 1,
             "N/A": 2}
        )
        label_vocab.create()

        self.tokenizer = AutoTokenizer.from_pretrained(plm_name)
        self.device = device

        self.plm = PlmAutoModel(plm_name)
        self.model = BaseTextClassificationModel(plm=self.plm,vocab={"label_vocab":label_vocab})
        load_model(self.model,model_path)
        self.model.to(self.device)

        self.type2type = {}
        self.type2type[int] = torch.long
        self.type2type[float] = torch.float
        self.type2type[bool] = torch.bool

        self.max_token_len = 128

    def run(self,context,response):
        tokenized_data = self.tokenizer.encode_plus(text=context,
                                                    text_pair=response,
                                                    padding="max_length",
                                                    add_special_tokens=True,
                                                    truncation=True,
                                                    max_length=self.max_token_len)

        input_dict = self.convert_to_sensor(tokenized_data)
        move_dict_value_to_device(input_dict,self.device)
        with torch.no_grad():
            pred = self.model.forward(input_dict)
            logits = F.softmax(pred, dim=1).cpu().numpy()
        return logits

    def convert_to_sensor(self,dict_data):
        """

        :param dict_data:
        :return:
        """
        new_tensor_data = {}
        for key,value in dict_data.items():
            single_element = value[0]
            if type(single_element) in self.type2type:
                dtype = self.type2type[type(single_element)]
                new_tensor_data[key] = torch.unsqueeze(torch.tensor(value,dtype=dtype),dim=0)
        return new_tensor_data

if __name__ == '__main__':
    toolkit = DialogeSafetyDetectionToolkit(
        plm_name='roberta-base',
        model_path='/data/hongbang/CogAGENT/datapath/dialogue_safety/DiaSafety/experiment/train_agreement--2023-02-19--06-04-08.48/best_model/checkpoint-2700/models.pt',
        device=torch.device("cuda:1"),
    )
    context = "I'm a huge fan of this nutritionally balanced dish, and I don't think vegans are strong enough."
    response = "Yeah."
    logits = toolkit.run(context,response)
    # sentence = "Downald trump is not doing well recently."
    # label = toolkit.run(sentence)
    # print("label:",label)
    from detoxify import Detoxify
    results = Detoxify('unbiased').predict(["Vegetarians eat that stuff, it's definitely not strong."])


