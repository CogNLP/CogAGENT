from cogagent.models.base_model import BaseModel
from cogagent.modules.diffks_modules import MyGRU
import torch.nn as nn
import torch.nn.functional as F
import torch
from cotk.wordvector import Glove
import numpy as np

class DiffKSModel(BaseModel):
    def __init__(self, vocab, glove_path, embedding_size=300,
                 eh_size=200,dh_size=400,hist_len=2,hist_weights=np.array([0.7,0.3]),
                 drop_out=0.5):
        super().__init__()
        self.vocab = vocab

        vocab_size = len(vocab["vocab_list"])
        unk_id = self.vocab["word_vocab"].label2id("<unk>")

        self.embLayer = EmbeddingLayer(vocab_size, embedding_size, drop_out,
                                       glove_path=glove_path, vocab_list=vocab["vocab_list"])
        self.postEncoder = PostEncoder(embedding_size,eh_size)
        self.wikiEncoder = WikiEncoder(embedding_size,eh_size)
        self.connectLayer = ConnectLayer(eh_size,dh_size,hist_len,hist_weights)
        self.genNetwork = GenNetwork(embedding_size,eh_size,dh_size,vocab_size,unk_id,drop_out)

    def loss(self, batch, loss_function):
        pred = self.forward(batch)

        dialog_turns = batch["post"].shape[1]
        for i in range(dialog_turns):
            pass

    def forward(self, batch):
        post,resp,wiki = batch["post"],batch["resp"],batch["wiki"]
        post_length,resp_length = batch["post_length"],batch["resp_length"]
        wiki_length = batch["wiki_length"]
        dialogue_turns = post.shape[1]
        for i in range(dialogue_turns):
            curr_post,curr_resp,curr_wiki = post[:,i,:],resp[:,i,:],wiki[:,i,:,:]
            curr_post_length,curr_resp_length = post_length[:,i], resp_length[:,i]
            curr_wiki_length = wiki_length[:,i]
            curr_post_embedding,curr_resp_embedding,curr_wiki_embedding = self.embLayer(curr_post),\
                                                           self.embLayer(curr_resp),\
                                                           self.embLayer(curr_wiki)
            h,hn  = self.postEncoder(curr_post_embedding,curr_post_length)
            h1,hn1 = self.wikiEncoder(curr_wiki_embedding,curr_wiki_length)


            print("Debug Usage")


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_size, dropout=0.5, glove_path=None, vocab_list=None):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        if glove_path:
            word_vector = Glove(glove_path)
            self.embedding.weight = nn.Parameter(torch.Tensor(word_vector.load_matrix(embedding_size, vocab_list)))

    def forward(self,data):
        """
        embedding layer
        :param data: (batch_size,*) (8,200,100,)
        :return: (batch_size,*,embedding_size) （8,200,100,300)
        """
        return self.dropout(self.embedding(data))


class PostEncoder(nn.Module):
    def __init__(self, embedding_size, eh_size):
        super(PostEncoder, self).__init__()
        self.postGRU = MyGRU(embedding_size, eh_size, bidirectional=True)

    def forward(self,raw_post,raw_post_length):
        """
        Encode the utterance and system response
        :param post:(batch_size,utterance_sequence_length,embedding_size) (8,100,300)
        :param post_length:(batch_size,)
        :return:
        """
        valid_sen =torch.sum(torch.nonzero(raw_post_length),1) # (batch_size,)
        raw_reverse = torch.cumsum(torch.gt(raw_post_length, 0), 0) - 1 # (batch_size)
        valid_num = valid_sen.shape[0]

        post = torch.index_select(raw_post, 0, valid_sen).transpose(0, 1)  # (utterance_sequence_length, valid_num, embedding_dim)
        post_length = torch.index_select(raw_post_length, 0, valid_sen)
        post_length = post_length.cpu().numpy()

        h,hn = self.postGRU.forward(post, post_length, need_h=True)
        return h,hn



class WikiEncoder(nn.Module):
    def __init__(self, embedding_size, eh_size):
        super(WikiEncoder, self).__init__()
        self.sentenceGRU = MyGRU(embedding_size, eh_size, bidirectional=True)

    def forward(self,wiki,wiki_length):
        """
        encode wiki passages
        :param wiki: (batch_size,wiki_num,wiki_sentence_length,embedding_dim)
        :param wiki_length: (batch_size,wiki_num)
        :return:
        """
        batch_size = wiki.shape[0]
        wiki_length = wiki_length.reshape(-1) # (batch_size * wiki_num)
        embed = wiki.reshape((-1,wiki.shape[2],wiki.shape[3])) # (batch_size * wiki_num,wiki_sentence_length,embedding_dim)
        embed = embed.transpose(0,1) # (wiki_sentence_length,batch_size * wiki_num,embedding_dim)

        # (wiki_sentence_length, batch * wiki_num, 2 * eh_size), (batch * wiki_num, 2 * eh_size)
        h1,h_n1 = self.sentenceGRU.forward(embed,wiki_length.cpu().numpy(),need_h=True)

        h1 = h1.reshape((h1.shape[0],batch_size,-1,h1.shape[-1])) # (wiki_sentence_length, batch_size, wiki_sen_num, 2 * eh_size)
        h_n1 = h_n1.reshape((batch_size,-1,h_n1.shape[-1])) # (batch_size,wiki_num,wiki_sentence_length,embedding_dim)
        return h1,h_n1


class ConnectLayer(nn.Module):
    def __init__(self, eh_size,dh_size,hist_len,hist_weights,):
        super(ConnectLayer, self).__init__()
        self.initLinearLayer = nn.Linear(eh_size * 4,dh_size)
        self.wiki_atten = nn.Softmax(dim=0)
        self.atten_lossCE = nn.CrossEntropyLoss(ignore_index=0)

        self.last_wiki = None
        self.hist_len = hist_len
        self.hist_weights = hist_weights

        self.compareGRU = MyGRU(2 * eh_size, eh_size, bidirectional=True)

        self.tilde_linear = nn.Linear(4 * eh_size, 2 * eh_size)
        self.attn_query = nn.Linear(2 * eh_size, 2 * eh_size, bias=False)
        self.attn_key = nn.Linear(4 * eh_size, 2 * eh_size, bias=False)
        self.attn_v = nn.Linear(2 * eh_size, 1, bias=False)

    def forward(self):
        pass



class GenNetwork(nn.Module):
    def __init__(self, embedding_size,eh_size,dh_size,vocab_size,unk_id,droprate):
        super(GenNetwork, self).__init__()
        self.GRULayer = MyGRU(embedding_size + 2 * eh_size, dh_size, initpara=False)
        self.wLinearLayer = nn.Linear(dh_size, vocab_size)
        self.lossCE = nn.NLLLoss(ignore_index=unk_id)
        self.wCopyLinear = nn.Linear(eh_size * 2, dh_size)
        self.drop = nn.Dropout(droprate)
        self.start_generate_id = 2

