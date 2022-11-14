from cogagent.toolkits.base_toolkit import BaseToolkit
from cogagent.data.processors.sst2_processors.sst2_processor import text_classification_for_sst2
from cogagent.models.base_text_classification_model import BaseTextClassificationModel
from cogagent.models.plms.plm_auto import PlmAutoModel
from collections import defaultdict
from cogagent.utils.train_utils import move_dict_value_to_device
from cogagent.models.diffks_model import DiffKSModel
from cogagent import load_model,load_pickle

import torch

class KnowledgeGroundedConversationToolkit(BaseToolkit):
    def __init__(self,bert_model,model_path,vocabulary_path,device):
        super(KnowledgeGroundedConversationToolkit, self).__init__(bert_model,model_path,vocabulary_path,device)
        if bert_model:
            self.plm = PlmAutoModel(bert_model)
        self.topic2knowledge = load_pickle("/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/wow/cache/topic2knowledge.pkl")
        vocab = {"word_vocab":self.vocabulary}
        self.model = DiffKSModel(vocab,glove_path=None)
        self.model = load_model(self.model,self.model_path)
        self.model.to(self.device)
        self.type2type = {}
        self.type2type[int] = torch.long
        self.type2type[float] = torch.float
        self.type2type[bool] = torch.bool

    def run(self,sentence):
        tokenized_data = text_classification_for_sst2(sentence,tokenizer=self.tokenizer,max_token_len=256)
        input_dict = self.convert_to_sensor(tokenized_data)
        move_dict_value_to_device(input_dict,self.device)
        label_id = self.model.predict(input_dict)
        label = self.vocabulary.id2label(label_id.clone().detach().cpu().item())
        return label

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
    toolkit = KnowledgeGroundedConversationToolkit(
        bert_model=None,
        model_path='/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/wow/experimental_result/run_diffks_wow--2022-11-13--04-17-09.20/best_model/checkpoint-9215/models.pt',
        vocabulary_path='/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/wow/cache/wow_vocab.pkl',
        device=torch.device("cuda:0"),
        )
    # sentence = "Downald trump is not doing well recently."
    # label = toolkit.run(sentence)
    # print("label:",label)


