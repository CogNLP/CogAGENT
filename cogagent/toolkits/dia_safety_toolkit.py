import os
import numpy as np
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
    def __init__(self,plm_name,classifier_path,device):
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

        self.classifiers = {}
        self.categories = ['agreement', 'expertise', 'offend','bias','risk']
        for category in self.categories:
            plm = PlmAutoModel(plm_name)
            model = BaseTextClassificationModel(plm=plm, vocab={"label_vocab": label_vocab})
            model_path = os.path.join(classifier_path,"model_"+category+".pt")
            load_model(model, model_path)
            model.to(device)
            self.classifiers[category] = model

        # self.plm = PlmAutoModel(plm_name)
        # self.model = BaseTextClassificationModel(plm=self.plm,vocab={"label_vocab":label_vocab})
        # load_model(self.model,model_path)
        # self.model.to(self.device)

        self.type2type = {}
        self.type2type[int] = torch.long
        self.type2type[float] = torch.float
        self.type2type[bool] = torch.bool

        self.max_token_len = 128

    def single_category_run(self,context,response,category):
        """
        run a single category safety detection
        :param context: user input
        :param response: agent response
        :param category: which classifier to use
        :return: logits (safe_prob,unsafe_prob,NA_prob) NA:not applicable
        """
        assert category in self.categories
        tokenized_data = self.tokenizer.encode_plus(text=context,
                                                    text_pair=response,
                                                    padding="max_length",
                                                    add_special_tokens=True,
                                                    truncation=True,
                                                    max_length=self.max_token_len)

        input_dict = self.convert_to_sensor(tokenized_data)
        move_dict_value_to_device(input_dict,self.device)
        with torch.no_grad():
            pred = self.classifiers[category].forward(input_dict)
            logits = F.softmax(pred, dim=1).cpu().numpy()
        return logits

    def run(self,context,response):
        results = {}
        for category in self.categories:
            results[category] = self.single_category_run(context,response,category)
        logits = np.stack([v.squeeze() for k,v in results.items()],axis=0)
        index = np.argmax(logits,axis=1)  # (5,)
        safe = np.sum(index == 1) == 0 # all the label is not unsafe!

        # choose the max probs among the unsafe predictions
        category = self.categories[np.where(index == 1)[0][np.argmax(logits[index==1,1])]] if not safe else 'safe'
        return safe,category


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
        classifier_path='/data/hongbang/CogAGENT/datapath/dialogue_safety/DiaSafety/raw_data/classifiers',
        device=torch.device("cuda:1"),
    )
    context = "I think Vegans are hypocritical"
    response = "Yeah."
    safe,category = toolkit.run(context,response)
    print("Safety:",safe,"  Category:",category)
    print("End")

    # sentence = "Downald trump is not doing well recently."
    # label = toolkit.run(sentence)
    # print("label:",label)
    # from detoxify import Detoxify
    # results = Detoxify('unbiased').predict(["Vegetarians eat that stuff, it's definitely not strong."])


