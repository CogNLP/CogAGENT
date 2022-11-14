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
        self.disentangle = True

        # vocab_size = len(vocab["vocab_list"])
        vocab_size = len(vocab["word_vocab"])
        unk_id = self.vocab["word_vocab"].label2id("<unk>")
        self.max_sent_length = 512

        self.embLayer = EmbeddingLayer(vocab_size, embedding_size, drop_out,
                                       glove_path=glove_path, vocab_list=[id for id in self.vocab["word_vocab"].id2label_dict.values()])
        self.postEncoder = PostEncoder(embedding_size,eh_size)
        self.wikiEncoder = WikiEncoder(embedding_size,eh_size)
        self.connectLayer = ConnectLayer(eh_size,dh_size,hist_len,hist_weights)
        self.genNetwork = GenNetwork(embedding_size,eh_size,dh_size,vocab_size,unk_id,drop_out,vocab)

    def loss(self, batch, loss_function):
        sent_loss,atten_loss = self.forward(batch)

        return sent_loss + atten_loss

    def evaluate(self, batch, metric_function):
        resp = batch["resp"]
        sent_num = batch["sent_num"]
        resp_all_vocabs = batch["resp_all_vocabs"]
        w_o_all = self.detail_forward(batch)
        metric_function.evaluate(w_o_all,resp_all_vocabs,sent_num)

    def detail_forward(self,batch):
        """
        forward process for evaluation
        :param batch:
        :return:
        """
        post,resp,wiki = batch["post"],batch["resp"],batch["wiki"]
        post_length,resp_length = batch["post_length"],batch["resp_length"]
        wiki_length = batch["wiki_length"]
        device = batch["post"].device
        dialogue_turns = torch.max(batch["sent_num"]).item()
        dm = self.vocab
        go_id = self.vocab["word_vocab"].label2id("<go>")
        eos_id = self.vocab["word_vocab"].label2id("<eos>")
        pad_id = self.vocab["word_vocab"].label2id("<pad>")

        def pad_post(posts):
            '''
            :param posts: list, [batch, turn_length, sent_length]
            '''
            post_length = np.array(list(map(len, posts)), dtype=int)
            post = np.zeros((len(post_length), np.max(post_length)), dtype=int)
            for j in range(len(post_length)):
                post[j, :len(posts[j])] = posts[j]
            return post, post_length

        ori_post = [[single_post[single_post != pad_id].tolist() for single_post in posts] for posts in post]
        new_post = [each[0] for each in ori_post]

        for i in range(dialogue_turns):
            state_num = i
            curr_post,curr_resp,curr_wiki = post[:,i,:],resp[:,i,:],wiki[:,i,:,:]
            curr_post_length,curr_resp_length = post_length[:,i], resp_length[:,i]
            curr_wiki_length = wiki_length[:,i]
            curr_wiki_num = batch["wiki_num"][:,i]
            curr_post,curr_post_length = [torch.from_numpy(elem).to(device) for elem in pad_post(new_post)]
            curr_post_embedding,curr_resp_embedding,curr_wiki_embedding = self.embLayer.detail_forward(curr_post),\
                                                           self.embLayer.detail_forward(curr_resp),\
                                                           self.embLayer.detail_forward(curr_wiki)
            h, hn, valid_sen, reverse_valid_sen = self.postEncoder.detail_forward(curr_post_embedding, curr_post_length)
            h1, hn1 = self.wikiEncoder(curr_wiki_embedding, curr_wiki_length)
            if not self.disentangle:
                selected_wiki_sen,selected_wiki_h,init_h,wiki_cv,single_atten_loss,(acc_prob,acc_label,acc_pred) = self.connectLayer.detail_forward()
            else:
                selected_wiki_sen,selected_wiki_h,init_h,wiki_cv,single_atten_loss,(acc_prob,acc_label,acc_pred) = self.connectLayer.detail_forward_disentangle(
                    i,valid_sen,reverse_valid_sen,curr_wiki,curr_wiki_num,h1,hn1,batch["atten"][:,i],h,hn
                )
            w_o_all = self.genNetwork.detail_forward(i,valid_sen,reverse_valid_sen,selected_wiki_sen,selected_wiki_h,
                                    curr_resp,curr_resp_length,curr_resp_embedding,init_h,wiki_cv,self.embLayer.embedding)


            if i < dialogue_turns - 1:
                gen_resp = w_o_all[-1].transpose(0, 1).cpu().tolist()
                new_post = []
                for j, gr in enumerate(gen_resp):
                    gr = np.array(gr)
                    new_post.append((ori_post[j][i] + ([go_id] + gr[gr!=pad_id].tolist() + [eos_id])[:self.max_sent_length] + ori_post[j][i + 1])[-100:])

        return w_o_all

    def forward(self, batch):
        post,resp,wiki = batch["post"],batch["resp"],batch["wiki"]
        post_length,resp_length = batch["post_length"],batch["resp_length"]
        wiki_length = batch["wiki_length"]
        dialogue_turns = torch.max(batch["sent_num"]).item()
        atten_loss = torch.zeros(1,device=post.device)
        sen_loss = torch.zeros(1,device=post.device)
        for i in range(dialogue_turns):
            curr_post,curr_resp,curr_wiki = post[:,i,:],resp[:,i,:],wiki[:,i,:,:]
            curr_post_length,curr_resp_length = post_length[:,i], resp_length[:,i]
            curr_wiki_length = wiki_length[:,i]
            curr_wiki_num = batch["wiki_num"][:,i]
            curr_post_embedding,curr_resp_embedding,curr_wiki_embedding = self.embLayer(curr_post),\
                                                           self.embLayer(curr_resp),\
                                                           self.embLayer(curr_wiki)
            h,hn,valid_sen,reverse_valid_sen = self.postEncoder(curr_post_embedding,curr_post_length)
            h1,hn1 = self.wikiEncoder(curr_wiki_embedding,curr_wiki_length)
            if not self.disentangle:
                selected_wiki_sen,selected_wiki_h,init_h,wiki_cv = self.connectLayer.forward()
            else:
                selected_wiki_sen,selected_wiki_h,init_h,wiki_cv,single_atten_loss = self.connectLayer.forward_disentangle(
                    i,valid_sen,reverse_valid_sen,curr_wiki,curr_wiki_num,h1,hn1,batch["atten"][:,i],h,hn
                )
                atten_loss += single_atten_loss

            single_sen_loss = self.genNetwork.forward(i,valid_sen,reverse_valid_sen,selected_wiki_sen,selected_wiki_h,
                                    curr_resp,curr_resp_length,curr_resp_embedding,init_h,wiki_cv)
            sen_loss += single_sen_loss
        return sen_loss,atten_loss

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

    def detail_forward(self,data):
        return self.embedding(data)


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
        reverse_valid_sen = raw_reverse * torch.ge(raw_reverse, 0).to(torch.long) # (batch_size)
        valid_num = valid_sen.shape[0]

        post = torch.index_select(raw_post, 0, valid_sen).transpose(0, 1)  # (utterance_sequence_length, valid_num, embedding_dim)
        post_length = torch.index_select(raw_post_length, 0, valid_sen)
        post_length = post_length.cpu().numpy()

        h,hn = self.postGRU.forward(post, post_length, need_h=True)
        return h,hn,valid_sen,reverse_valid_sen

    def detail_forward(self,raw_post,raw_post_length):
        valid_sen =torch.sum(torch.nonzero(raw_post_length),1) # (batch_size,)
        raw_reverse = torch.cumsum(torch.gt(raw_post_length, 0), 0) - 1 # (batch_size)
        reverse_valid_sen = raw_reverse * torch.ge(raw_reverse, 0).to(torch.long) # (batch_size)
        valid_num = valid_sen.shape[0]

        post = torch.index_select(raw_post, 0, valid_sen).transpose(0, 1)  # (utterance_sequence_length, valid_num, embedding_dim)
        post_length = torch.index_select(raw_post_length, 0, valid_sen)
        post_length = post_length.cpu().numpy()

        h,hn = self.postGRU.forward(post, post_length, need_h=True)
        return h,hn,valid_sen,reverse_valid_sen



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
        h_n1 = h_n1.reshape((batch_size,-1,h_n1.shape[-1])).transpose(0, 1) # (wiki_num,batch_size,wiki_sentence_length,2 * eh_size)
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
        return None,None,None,None

    def detail_forward(self):
        return None,None,None,None,None,[None,None,None]

    def forward_disentangle(self,i,valid_sen,reverse_valid_sen,wiki_sen,wiki_num,h1,h_n1,atten_label,h,h_n):
        valid_wiki_h_n1 = torch.index_select(h_n1, 1, valid_sen)
        valid_wiki_sen = torch.index_select(wiki_sen, 0, valid_sen)
        valid_wiki_h1 = torch.index_select(h1,1,valid_sen)
        atten_label = torch.index_select(atten_label,0,valid_sen)
        valid_wiki_num = torch.index_select(wiki_num,0,valid_sen)

        self.beta = torch.sum(valid_wiki_h_n1 * h_n, dim = 2).T # (batch_size_valid,wiki_len)
        mask = torch.arange(self.beta.shape[1], device=self.beta.device).long().expand(self.beta.shape[0],self.beta.shape[1]).transpose(0,1)
        expand_wiki_num = valid_wiki_num.unsqueeze(0).expand_as(mask)
        reverse_mask = (expand_wiki_num <= mask).float()

        if i > 0:
            h2,hn_2 = self.compareGRU(h_n1,wiki_num.cpu().numpy(),need_h=True)
            valid_wiki_h2 = torch.index_select(h2, 1, valid_sen) # (wiki_len,batch_valid_num,2 * eh_size)

            tilde_wiki_list = []
            for i in range(self.last_wiki.size(-1)):
                last_wiki = torch.index_select(self.last_wiki[:, :, i], 0, valid_sen).unsqueeze(
                    0)  # 1, valid_num, (2 * eh)
                tilde_wiki = torch.tanh(
                    self.tilde_linear(torch.cat([last_wiki - valid_wiki_h2, last_wiki * valid_wiki_h2], dim=-1)))
                tilde_wiki_list.append(tilde_wiki.unsqueeze(-1) * self.hist_weights[i])
            tilde_wiki = torch.cat(tilde_wiki_list, dim=-1).sum(dim=-1)
            # wiki_len * valid_num * (2 * eh_size)

            query = self.attn_query(tilde_wiki)  # [1, valid_num, hidden]
            key = self.attn_key(torch.cat([valid_wiki_h2, tilde_wiki], dim=-1))  # [wiki_sen_num, valid_num, hidden]
            atten_sum = self.attn_v(torch.tanh(query + key)).squeeze(-1)  # [wiki_sen_num, valid_num]

            self.beta = self.beta[:, :atten_sum.shape[0]] + torch.t(atten_sum)

        atten_loss = self.atten_lossCE(self.beta,atten_label)
        atten_loss = torch.zeros_like(atten_loss) if torch.isnan(atten_loss).detach().cpu().numpy() > 0 else atten_loss

        golden_alpha = torch.zeros_like(self.beta).scatter_(1, atten_label.unsqueeze(1), 1)
        golden_alpha = torch.t(golden_alpha).unsqueeze(2) # (wiki_num,batch_valid_num,1)
        wiki_cv = torch.sum(valid_wiki_h_n1[:golden_alpha.shape[0]] * golden_alpha, dim=0) # (batch_valid_num,2*eh_size)
        init_h = self.initLinearLayer(torch.cat([h_n, wiki_cv], 1)) # (batch_valid_num,2*eh_size)

        if i == 0:
            self.last_wiki = torch.index_select(wiki_cv, 0, reverse_valid_sen).unsqueeze(-1)  # [batch, 2 * eh_size]
        else:
            self.last_wiki = torch.cat([torch.index_select(wiki_cv, 0, reverse_valid_sen).unsqueeze(-1),
                                        self.last_wiki[:, :, :self.hist_len-1]], dim=-1)

        atten_indices = atten_label.unsqueeze(1) # (batch_valid_num,1)
        atten_indices = torch.cat([torch.arange(atten_indices.shape[0]).unsqueeze(1), atten_indices.cpu()], 1) # (batch_valid_num,1)
        valid_wiki_h1 = torch.transpose(valid_wiki_h1, 0, 1) # (batch_valid_num, wiki_len, wiki_num,(2 * eh_size))
        valid_wiki_h1 = torch.transpose(valid_wiki_h1, 1, 2) # (batch_valid_num, wiki_num ,wiki_sen_len, (2 * eh_size))
        selected_wiki_h = valid_wiki_h1[atten_indices.chunk(2, 1)].squeeze(1)
        selected_wiki_sen = valid_wiki_sen[atten_indices.chunk(2, 1)].squeeze(1)

        return selected_wiki_sen,selected_wiki_h,init_h,wiki_cv,atten_loss

    def detail_forward_disentangle(self,i,valid_sen,reverse_valid_sen,wiki_sen,wiki_num,h1,h_n1,atten_label,h,h_n):
        valid_wiki_h_n1 = torch.index_select(h_n1, 1, valid_sen)
        valid_wiki_sen = torch.index_select(wiki_sen, 0, valid_sen)
        valid_wiki_h1 = torch.index_select(h1,1,valid_sen)
        atten_label = torch.index_select(atten_label,0,valid_sen)
        valid_wiki_num = torch.index_select(wiki_num,0,valid_sen)

        self.beta = torch.sum(valid_wiki_h_n1 * h_n, dim = 2).T # (batch_size_valid,wiki_len)
        mask = torch.arange(self.beta.shape[1], device=self.beta.device).long().expand(self.beta.shape[0],self.beta.shape[1]).transpose(0,1)
        expand_wiki_num = valid_wiki_num.unsqueeze(0).expand_as(mask)
        reverse_mask = (expand_wiki_num <= mask).float()

        if i > 0:
            h2,hn_2 = self.compareGRU(h_n1,wiki_num.cpu().numpy(),need_h=True)
            valid_wiki_h2 = torch.index_select(h2, 1, valid_sen) # (wiki_len,batch_valid_num,2 * eh_size)

            tilde_wiki_list = []
            for i in range(self.last_wiki.size(-1)):
                last_wiki = torch.index_select(self.last_wiki[:, :, i], 0, valid_sen).unsqueeze(
                    0)  # 1, valid_num, (2 * eh)
                tilde_wiki = torch.tanh(
                    self.tilde_linear(torch.cat([last_wiki - valid_wiki_h2, last_wiki * valid_wiki_h2], dim=-1)))
                tilde_wiki_list.append(tilde_wiki.unsqueeze(-1) * self.hist_weights[i])
            tilde_wiki = torch.cat(tilde_wiki_list, dim=-1).sum(dim=-1)
            # wiki_len * valid_num * (2 * eh_size)

            query = self.attn_query(tilde_wiki)  # [1, valid_num, hidden]
            key = self.attn_key(torch.cat([valid_wiki_h2, tilde_wiki], dim=-1))  # [wiki_sen_num, valid_num, hidden]
            atten_sum = self.attn_v(torch.tanh(query + key)).squeeze(-1)  # [wiki_sen_num, valid_num]

            self.beta = self.beta[:, :atten_sum.shape[0]] + torch.t(atten_sum)

        atten_loss = self.atten_lossCE(self.beta,atten_label)
        atten_loss = torch.zeros_like(atten_loss) if torch.isnan(atten_loss).detach().cpu().numpy() > 0 else atten_loss

        self.beta = torch.t(self.beta) - 1e10 * reverse_mask[:self.beta.shape[1]]
        self.alpha = self.wiki_atten(self.beta)  # wiki_len * valid_num
        acc_prob = torch.index_select(self.alpha.t(), 0, reverse_valid_sen).cpu().tolist()
        atten_indices = torch.argmax(self.alpha, 0) # valid_num
        alpha =  torch.zeros_like(self.beta.T).scatter_(1, atten_indices.unsqueeze(1), 1)
        alpha = alpha.T

        wiki_cv = torch.sum(valid_wiki_h_n1[:alpha.shape[0]] * alpha.unsqueeze(2), dim=0) # (batch_valid_num,2*eh_size)
        init_h = self.initLinearLayer(torch.cat([h_n, wiki_cv], 1)) # (batch_valid_num,2*eh_size)

        if i == 0:
            self.last_wiki = torch.index_select(wiki_cv, 0, reverse_valid_sen).unsqueeze(-1)  # [batch, 2 * eh_size]
        else:
            self.last_wiki = torch.cat([torch.index_select(wiki_cv, 0, reverse_valid_sen).unsqueeze(-1),
                                        self.last_wiki[:, :, :self.hist_len-1]], dim=-1)

        acc_label = torch.index_select(atten_label, 0, reverse_valid_sen).cpu().tolist()
        acc_pred = torch.index_select(atten_indices, 0, reverse_valid_sen).cpu().tolist()

        atten_indices = atten_indices.unsqueeze(1)
        atten_indices = torch.cat([torch.arange(atten_indices.shape[0]).unsqueeze(1), atten_indices.cpu()], 1) # (batch_valid_num,1)
        valid_wiki_h1 = torch.transpose(valid_wiki_h1, 0, 1) # (batch_valid_num, wiki_len, wiki_num,(2 * eh_size))
        valid_wiki_h1 = torch.transpose(valid_wiki_h1, 1, 2) # (batch_valid_num, wiki_num ,wiki_sen_len, (2 * eh_size))
        selected_wiki_h = valid_wiki_h1[atten_indices.chunk(2, 1)].squeeze(1)
        selected_wiki_sen = valid_wiki_sen[atten_indices.chunk(2, 1)].squeeze(1)

        return selected_wiki_sen,selected_wiki_h,init_h,wiki_cv,atten_loss,[acc_prob,acc_label,acc_pred]


class GenNetwork(nn.Module):
    def __init__(self, embedding_size,eh_size,dh_size,vocab_size,unk_id,droprate,vocab):
        super(GenNetwork, self).__init__()
        self.GRULayer = MyGRU(embedding_size + 2 * eh_size, dh_size, initpara=False)
        self.wLinearLayer = nn.Linear(dh_size, vocab_size)
        self.lossCE = nn.NLLLoss(ignore_index=unk_id)
        self.wCopyLinear = nn.Linear(eh_size * 2, dh_size)
        self.drop = nn.Dropout(droprate)
        self.start_generate_id = 2
        self.vocab_size = vocab_size
        self.max_sent_length = 512
        self.vocab = vocab

        self.w_o_all = None

    def forward(self,i,valid_sen,reverse_valid_sen,
                selected_wiki_sen,selected_wiki_h,
                curr_resp,curr_resp_length,curr_resp_embedding,
                init_h,wiki_cv):
        resp_length = torch.index_select(curr_resp_length,0,valid_sen).cpu().numpy()
        resp_embedding = torch.index_select(curr_resp_embedding,0,valid_sen).transpose(0, 1)
        resp = torch.index_select(curr_resp,0,valid_sen).transpose(0, 1)[1:]
        gen_w,gen_p = self.teacherForcing(resp_embedding,resp_length,wiki_cv,init_h,selected_wiki_sen,selected_wiki_h)

        # gen.h_n: valid_num * dh_dim

        w_slice = torch.index_select(gen_w, 1, reverse_valid_sen)
        if w_slice.shape[0] < self.max_sent_length:
            w_slice = torch.cat([w_slice, torch.zeros(self.max_sent_length - w_slice.shape[0], w_slice.shape[1], w_slice.shape[2],device=w_slice.device)], 0)
        if i == 0:
            self.incoming_state_w_all = w_slice.unsqueeze(0)
        else:
            self.incoming_state_w_all = torch.cat([self.incoming_state_w_all, w_slice.unsqueeze(0)], 0) #state.w_all: sen_num * sen_length * batch_size * vocab_size

        w_o_f = flattenSequence(torch.log(gen_p), resp_length-1)
        data_f = flattenSequence(resp, resp_length-1)
        # incoming.statistic.sen_num += incoming.state.valid_num
        now = 0
        incoming_result_word_loss = None
        for l in resp_length:
            loss = self.lossCE(w_o_f[now:now+l-1, :], data_f[now:now+l-1])
            if incoming_result_word_loss is None:
                incoming_result_word_loss = loss.clone()
            else:
                incoming_result_word_loss += loss.clone()
            # incoming.statistic.sen_loss.append(loss.item())
            now += l - 1
        return incoming_result_word_loss
        # if i == incoming.state.last - 1:
        #     incoming.statistic.sen_loss = torch.tensor(incoming.statistic.sen_loss)
        #     incoming.result.perplexity = torch.mean(torch.exp(incoming.statistic.sen_loss))


    def teacherForcing(self,embedding,length,wiki_cv,init_h,selected_wiki_sen,selected_wiki_h):
        wiki_cv = wiki_cv.unsqueeze(0).repeat(embedding.shape[0], 1, 1)

        gen_h, gen_h_n = self.GRULayer.forward(torch.cat([embedding, wiki_cv], dim=-1), length - 1, h_init=init_h,
                                               need_h=True)

        gen_w = self.wLinearLayer(self.drop(gen_h))
        gen_w = torch.clamp(gen_w, max=5.0)
        gen_vocab_p = torch.exp(gen_w)
        wikiState = torch.transpose(torch.tanh(self.wCopyLinear(selected_wiki_h)), 0, 1)
        copyW = torch.exp(torch.clamp(torch.unsqueeze(
            torch.transpose(torch.sum(torch.unsqueeze(gen_h, 1) * torch.unsqueeze(wikiState, 0), -1), 1, 2), 2),
                                      max=5.0))

        # selected_wiki_sen= selected_wiki_sen[:, :selected_wiki_sen.shape[1]]
        selected_wiki_sen = selected_wiki_sen[:, :copyW.shape[-1]]
        copyHead = torch.zeros(1, selected_wiki_sen.shape[0], selected_wiki_sen.shape[1],
                         self.vocab_size,dtype=torch.float32,device=selected_wiki_sen.device).scatter_(3,
                                                                     torch.unsqueeze(torch.unsqueeze(selected_wiki_sen, 0),
                                                                                     3), 1)
        gen_copy_p = torch.matmul(copyW, copyHead).squeeze(2)
        gen_p = gen_vocab_p + gen_copy_p + 1e-10
        gen_p = gen_p / torch.unsqueeze(torch.sum(gen_p, 2), 2)
        gen_p = torch.clamp(gen_p, 1e-10, 1.0)
        return gen_w,gen_p


    def detail_forward(self,i,valid_sen,reverse_valid_sen,
                selected_wiki_sen,selected_wiki_h,
                curr_resp,curr_resp_length,curr_resp_embedding,
                init_h,wiki_cv,embLayer):
        gen_w_o,gen_emb,gen_length,gen_h_n = self.free_run(i,valid_sen,reverse_valid_sen,
                selected_wiki_sen,selected_wiki_h,
                curr_resp,curr_resp_length,curr_resp_embedding,
                init_h,wiki_cv,embLayer)
        # dm = self.param.volatile.dm
        w_o = gen_w_o.detach().cpu().numpy()

        w_o_slice = torch.index_select(gen_w_o, 1, reverse_valid_sen)
        if w_o_slice.shape[0] < self.max_sent_length:
            w_o_slice = torch.cat([w_o_slice, torch.zeros(self.max_sent_length - w_o_slice.shape[0], w_o_slice.shape[1],device=gen_w_o.device,dtype=torch.long)], 0)

        if i == 0:
            self.w_o_all = w_o_slice.unsqueeze(0)
        else:
            self.w_o_all = torch.cat([self.w_o_all, w_o_slice.unsqueeze(0)], 0) #state.w_all: sen_num * sen_length * batch_size
        return self.w_o_all

    def free_run(self,i,valid_sen,reverse_valid_sen,
                selected_wiki_sen,selected_wiki_h,
                curr_resp,curr_resp_length,curr_resp_embedding,
                init_h,wiki_cv,embLayer,mode='max'):
        batch_size = len(valid_sen)
        device = next(embLayer.parameters()).device
        go_id = self.vocab["word_vocab"].label2id("<go>")
        eos_id = self.vocab["word_vocab"].label2id("<eos>")
        first_emb = embLayer(torch.tensor([go_id],
                                          dtype=torch.long,device=device)).repeat(batch_size,1)
        gen_w_pro = []
        gen_w_o = []
        gen_emb = []
        flag = torch.zeros(batch_size,device=device).byte()
        EOSmet = []

        selected_wiki_sen = selected_wiki_sen[:,:selected_wiki_h.shape[1]]
        copyHead = torch.zeros(1, selected_wiki_sen.shape[0], selected_wiki_sen.shape[1],
                    self.vocab_size, dtype=torch.float32, device=selected_wiki_sen.device).scatter_(3,
                                                                                    torch.unsqueeze(
                                                                                        torch.unsqueeze(
                                                                                            selected_wiki_sen,
                                                                                            0),
                                                                                3), 1)
        wikiState = torch.transpose(torch.tanh(self.wCopyLinear(selected_wiki_h)), 0, 1)
        next_emb = first_emb
        gru_h = init_h
        gen_p = []
        for _ in range(self.max_sent_length):
            now = torch.cat([next_emb, wiki_cv], dim=-1)

            gru_h = self.GRULayer.cell_forward(now, gru_h)
            w = self.wLinearLayer(gru_h)
            w = torch.clamp(w, max=5.0)
            vocab_p = torch.exp(w)
            copyW = torch.exp(
                torch.clamp(torch.unsqueeze((torch.sum(torch.unsqueeze(gru_h, 0) * wikiState, -1).transpose_(0, 1)), 1),
                            max=5.0))  # batch * 1 * wiki_len
            copy_p = torch.matmul(copyW, copyHead).squeeze()

            p = vocab_p + copy_p + 1e-10
            p = p / torch.unsqueeze(torch.sum(p, 1), 1)
            p = torch.clamp(p, 1e-10, 1.0)
            gen_p.append(p)

            if mode == "max":
                w_o = torch.argmax(p[:, self.start_generate_id:], dim=1) + self.start_generate_id
                next_emb = embLayer(w_o)
            elif mode == "gumbel":
                pass
            gen_w_o.append(w_o)
            gen_emb.append(next_emb)

            EOSmet.append(flag)
            flag = flag | (w_o == eos_id).byte()
            if torch.sum(flag).detach().cpu().numpy() == batch_size:
                break

        EOSmet = 1 - torch.stack(EOSmet)
        gen_w_o = torch.stack(gen_w_o) * EOSmet.long()
        gen_emb = torch.stack(gen_emb) * EOSmet.float().unsqueeze(-1)
        gen_length = torch.sum(EOSmet, 0).detach().cpu().numpy()
        gen_h_n = gru_h
        return gen_w_o,gen_emb,gen_length,gen_h_n


def flattenSequence(data, length):
    arr = []
    for i in range(length.size):
        arr.append(data[0:length[i], i])
    return torch.cat(arr, dim=0)
