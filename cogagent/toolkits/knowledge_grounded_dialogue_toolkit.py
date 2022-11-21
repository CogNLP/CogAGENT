from cogagent.toolkits.base_toolkit import BaseToolkit
from cogagent.data.processors.sst2_processors.sst2_processor import text_classification_for_sst2
from cogagent.models.base_text_classification_model import BaseTextClassificationModel
from cogagent.models.plms.plm_auto import PlmAutoModel
from collections import defaultdict
from cogagent.utils.train_utils import move_dict_value_to_device
from cogagent.models.diffks_model import DiffKSModel
from cogagent.utils.train_utils import move_dict_value_to_device
from cogagent import load_model,load_pickle
from random import shuffle
import random
from nltk.tokenize import WordPunctTokenizer
import numpy as np

import torch

class KnowledgeGroundedConversationAgent(BaseToolkit):
    def __init__(self,bert_model,model_path,vocabulary_path,device,debug=False):
        super(KnowledgeGroundedConversationAgent, self).__init__(bert_model,model_path,vocabulary_path,device)
        if bert_model:
            self.plm = PlmAutoModel(bert_model)
        self.topic2knowledge = load_pickle("/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/wow/cache/topic2knowledge.pkl")
        self.topics = list(self.topic2knowledge.keys())
        self.topic_choice_count = ['A','B','C','D','E']
        vocab = {"word_vocab":self.vocabulary}

        self.debug = debug

        self.max_token_len = 100
        self.word_vocab = self.vocabulary
        self.word2id= self.word_vocab.get_label2id_dict()
        self.id2word = self.word_vocab.get_id2label_dict()
        self.unk_id,self.eos_id = self.word2id["<unk>"],self.word2id["<eos>"]
        self.go_id,self.pad_id = self.word2id["<go>"],self.word2id["<pad>"]
        self.line2id = lambda line: ([self.go_id] + list(map(lambda word: self.word2id.get(word, self.unk_id), line)) + [self.eos_id])[:self.max_token_len]
        self.know2id = lambda line: (list(map(lambda word: self.word2id.get(word, self.unk_id), line)))[:self.max_token_len]
        self.id2line = lambda line: list(map(lambda id:self.id2word.get(id,"<unk>"),line))
        self.f = lambda sen: ' '.join(WordPunctTokenizer().tokenize(sen.strip())).lower()

        self.max_wiki_num = 280
        self.max_wiki_length = 256
        self.max_post_length = 128
        self.max_sent_length = 512
        self.valid_vocab_len = len(self.vocabulary)

        self.model = DiffKSModel(vocab,glove_path=None)
        self.model = load_model(self.model,self.model_path)
        self.model.to(self.device)
        self.type2type = {}
        self.type2type[int] = torch.long
        self.type2type[float] = torch.float
        self.type2type[bool] = torch.bool

    def run(self):
        topic = self.choose_topic()
        padded_wiki,wiki_length,wiki_num = self.search_knowledge(topic)
        dialogue_turns = int(padded_wiki.shape[0])

        chat_history = []

        input_msg = self.get_user_input()
        post = [self.line2id(self.f(input_msg).split())[:self.max_post_length]]

        for step in range(dialogue_turns):
            curr_post, curr_post_length = [elem for elem in pad_post(post)]
            input_batch = self.convert_to_sensor({
                "curr_post":curr_post[0].tolist(),
                "curr_post_length":curr_post_length[0].tolist(),
                "padded_wiki":padded_wiki.tolist(),
                "wiki_length":wiki_length,
                "wiki_num":wiki_num,
                "i":step,
            })
            move_dict_value_to_device(input_batch,self.device)
            gen_resp = self.model.predict(input_batch)
            gen_resp_str = " ".join(self.id2line(gen_resp[0]))
            print(">>Agent:",gen_resp_str)
            if step < dialogue_turns - 1:
                input_msg = self.get_user_input()
                gr = np.array(gen_resp[0])
                post = [(post[0] + ([self.go_id] + gr[gr!=self.pad_id].tolist() + [self.eos_id])[:self.max_sent_length] + self.line2id(self.f(input_msg).split())[:self.max_post_length])[-100:]]
            # break
        print(">>Agent:Reach max number of conversations!Bye!")

    def get_user_input(self,):
        if self.debug:
            input_msg = "Let's talk about it!"
            print(">>User:{}".format(input_msg))
        else:
            input_msg = input(">>User:")
        return input_msg

    def search_knowledge(self,topic):
        knowledge = self.topic2knowledge[topic]
        # wiki_str = max(knowledge,key=len)
        wiki_str = random.choice(knowledge)
        wiki = [list(map(self.know2id, wiki_psg)) for wiki_psg in wiki_str]
        max_sent_num = len(wiki)
        sent_num = max_sent_num
        f_wiki_num = lambda x: min(len(x), self.max_wiki_num)
        wiki_num = list(map(f_wiki_num, wiki))
        wiki_num += [0] * (max_sent_num - len(wiki_num))

        wiki_length = np.zeros((max_sent_num, self.max_wiki_num), dtype=int)
        for i in range(sent_num):
            f_wiki_length = lambda x: min(len(x), self.max_wiki_length)
            single_wiki_length = list(map(f_wiki_length, wiki[i]))[:self.max_wiki_num]
            wiki_length[i, :len(single_wiki_length)] = single_wiki_length

        wiki_length = wiki_length.tolist()

        padded_wiki = np.zeros((max_sent_num, self.max_wiki_num, self.max_wiki_length), dtype=int)
        for i in range(sent_num):
            # self.wiki_nums.append(wiki_num[i])
            for j in range(wiki_num[i]):
                # if wiki_num[i] > self.max_wiki_num:
                #     print("{} exceed the maximum wiki number {}!".format(wiki_num[i],self.max_wiki_num))
                single_wiki = wiki[i][j][:self.max_wiki_length]
                padded_wiki[i, j, :len(single_wiki)] = single_wiki

        padded_wiki[padded_wiki >= self.valid_vocab_len] = self.unk_id
        return padded_wiki,wiki_length,wiki_num

    def choose_topic(self):
        while True:
            print("Please choose one topic:")
            shuffle(self.topics)
            for i,choice in enumerate(self.topic_choice_count[:-1]):
                print("{}.{}".format(choice,self.topics[i]))
            print("{}.{}".format(self.topic_choice_count[-1],"I want to choose other topics"))
            if self.debug:
                choice = 'B'
            else:
                choice = input(">>")
            if choice not in self.topic_choice_count:
                print("You can only choose from {}!".format(" ".join(self.topic_choice_count)))
            elif choice == self.topic_choice_count[-1]:
                print("Shuffling...")
            else:
                return self.topics[self.topic_choice_count.index(choice)]



    # def run(self,sentence):
    #     tokenized_data = text_classification_for_sst2(sentence,tokenizer=self.tokenizer,max_token_len=256)
    #     input_dict = self.convert_to_sensor(tokenized_data)
    #     move_dict_value_to_device(input_dict,self.device)
    #     label_id = self.model.predict(input_dict)
    #     label = self.vocabulary.id2label(label_id.clone().detach().cpu().item())
    #     return label
    #
    def convert_to_sensor(self,dict_data):
        """

        :param dict_data:
        :return:
        """
        new_tensor_data = {}
        for key,value in dict_data.items():
            single_element = value
            while True:
                if isinstance(single_element,list):
                    single_element = single_element[0]
                else:
                    break
            if type(single_element) in self.type2type:
                dtype = self.type2type[type(single_element)]
                new_tensor_data[key] = torch.unsqueeze(torch.tensor(value,dtype=dtype),dim=0)
        return new_tensor_data

def pad_post(posts):
    '''
    :param posts: list, [batch, turn_length, sent_length]
    '''
    post_length = np.array(list(map(len, posts)), dtype=int)
    post = np.zeros((len(post_length), np.max(post_length)), dtype=int)
    for j in range(len(post_length)):
        post[j, :len(posts[j])] = posts[j]
    return post, post_length

if __name__ == '__main__':
    agent = KnowledgeGroundedConversationAgent(
        bert_model=None,
        model_path='/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/wow/experimental_result/run_diffks_wow_lr1e-4--2022-11-16--01-01-39.05/best_model/checkpoint-336000/models.pt',
        vocabulary_path='/data/hongbang/CogAGENT/datapath/knowledge_grounded_dialogue/wow/cache/wow_vocab.pkl',
        device=torch.device("cuda:0"),
        debug=False,
        )
    agent.run()

    # topic = agent.choose_topic()
    # print(topic)
    # sentence = "Downald trump is not doing well recently."
    # label = toolkit.run(sentence)
    # print("label:",label)


