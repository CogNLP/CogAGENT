from cogagent.models.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import AutoModel, AutoConfig
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1): # heads = 4, d_model = 768 dropout = 0
        super().__init__()

        self.d_model = d_model      # 768
        self.d_k = d_model // heads # 192
        self.h = heads              # 4

        self.q_linear = nn.Linear(d_model, d_model) # Linear(in_features=768, out_features=768, bias=True)
        self.v_linear = nn.Linear(d_model, d_model) # Linear(in_features=768, out_features=768, bias=True)
        self.k_linear = nn.Linear(d_model, d_model) # Linear(in_features=768, out_features=768, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)      # Linear(in_features=768, out_features=768, bias=True)

        self.scores = None

    def attention(self, q, k, v, d_k, mask=None, dropout=None):

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        self.scores = scores
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def get_scores(self):
        return self.scores

class Model(nn.Module):
    def __init__(self, config) -> None:
        super(Model, self).__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained('bert-base-uncased', config=self.config)
    def forward(self, input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False):
        result = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, encoder_hidden_states=output_all_encoded_layers)
        return result


class DstSumbtModel(BaseModel):
    def __init__(self, args, num_labels, device="cuda"):
        super(DstSumbtModel, self).__init__()

        self.hidden_dim = args.hidden_dim               # 300
        self.rnn_num_layers = args.num_rnn_layers       # 1
        self.zero_init_rnn = args.zero_init_rnn         # False
        self.max_seq_length = args.max_seq_length       # 64
        self.max_label_length = args.max_label_length   # 32
        self.num_labels = num_labels                    # [8, 146, 22, 2, 2, 5, 2, 52, 15, 9, 9, 9, 4, 93, ...]
        self.num_slots = len(num_labels)                # 35
        self.attn_head = args.attn_head                 # 4
        self.device = device

        self.acc_ = 0

        self.utterance_encoder = Model(AutoConfig.from_pretrained('bert-base-uncased'))
        self.utterance_encoder.train()
        self.bert_output_dim = self.utterance_encoder.config.hidden_size    # 768
        self.hidden_dropout_prob = self.utterance_encoder.config.hidden_dropout_prob    # 0.1
        if args.fix_utterance_encoder:
            for p in self.utterance_encoder.bert.pooler.parameters():
                p.requires_grad = False

        ### slot, slot-value Encoder (not trainable)
        # self.sv_encoder = BertForUtteranceEncoding.from_pretrained("bert-base-uncased", cache_dir=args.bert_model_cache_dir)
        # self.utterance_encoder = BertForUtteranceEncoding.from_pretrained(args.bert_model_name, cache_dir=args.bert_model_cache_dir)
        # from transformers import AutoConfig
        # self.sv_encoder = BertForUtteranceEncoding(AutoConfig.from_pretrained("bert-base-uncased"))
        self.sv_encoder = Model(AutoConfig.from_pretrained('bert-base-uncased'))
        self.sv_encoder.train()
        for p in self.sv_encoder.bert.parameters():
            p.requires_grad = False

        self.slot_lookup = nn.Embedding(self.num_slots, self.bert_output_dim) # Embedding(35, 768)
        self.value_lookup = nn.ModuleList([nn.Embedding(num_label, self.bert_output_dim) for num_label in num_labels])  #<bound method Module._named_members of ModuleList(  (0): Embedding(8, 768)  (1): Embedding(146, 768)  (2): Embedding(22, 768)  (3): Embedding(2, 768)  (4): Embedding(2, 768)  (5): Embedding(5, 768)  (6): Embedding(2, 768)  (7): Embedding(52, 768)  (8): Embedding(15, 768)  (9): Embedding(9, 768)  (10): Embedding(9, 768)  (11): Embedding(9, 768)  (12): Embedding(4, 768)  (13): Embedding(93, 768)  (14): Embedding(5, 768)  (15): Embedding(5, 768)  (16): Embedding(8, 768)  (17): Embedding(4, 768)  (18): Embedding(7, 768)  (19): Embedding(10, 768)  (20): Embedding(9, 768)  (21): Embedding(62, 768)  (22): Embedding(106, 768)  (23): Embedding(193, 768)  (24): Embedding(5, 768)  (25): Embedding(98, 768)  (26): Embedding(273, 768)  (27): Embedding(283, 768)  (28): Embedding(119, 768)  (29): Embedding(111, 768)  (30): Embedding(13, 768)  (31): Embedding(13, 768)  (32): Embedding(41, 768)  (33): Embedding(33, 768)  (34): Embedding(148, 768))>


        ### Attention layer
        self.attn = MultiHeadAttention(self.attn_head, self.bert_output_dim, dropout=0)

        ### RNN Belief Tracker
        self.nbt = None
        if args.task_name.find("gru") != -1:
            self.nbt = nn.GRU(input_size=self.bert_output_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=self.rnn_num_layers,
                              dropout=self.hidden_dropout_prob,
                              batch_first=True)
            self.init_parameter(self.nbt)
        elif args.task_name.find("lstm") != -1:
            self.nbt = nn.LSTM(input_size=self.bert_output_dim,
                               hidden_size=self.hidden_dim,
                               num_layers=self.rnn_num_layers,
                               dropout=self.hidden_dropout_prob,
                               batch_first=True)
            self.init_parameter(self.nbt)
        if not self.zero_init_rnn:
            self.rnn_init_linear = nn.Sequential(
                nn.Linear(self.bert_output_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.hidden_dropout_prob)
            )

        self.linear = nn.Linear(self.hidden_dim, self.bert_output_dim)
        self.layer_norm = nn.LayerNorm(self.bert_output_dim)

        ### Measure
        self.distance_metric = args.distance_metric
        if self.distance_metric == "cosine":
            self.metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
        elif self.distance_metric == "euclidean":
            self.metric = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

        ### Classifier
        self.nll = nn.CrossEntropyLoss(ignore_index=-1)

        ### Etc.
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # default evaluation mode
        self.eval()


    def forward(self, batch, n_gpu=1, target_slot=None):
         # input_ids = torch.Size([2, 22, 64]) | input_len = torch.Size([2, 22, 2]) | labels = torch.Size([2, 22, 35])
        # if target_slot is not specified, output values corresponding all slot-types
        input_ids, input_len, labels = batch['all_input_ids'], batch['all_input_len'], batch['all_label_ids']
        # input_ids, input_len, labels = batch
        if target_slot is None:
            target_slot = list(range(0, self.num_slots)) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, ...]

        ds = input_ids.size(0)  # dialog size  2
        ts = input_ids.size(1)  # turn size    22
        bs = ds * ts            # 44
        slot_dim = len(target_slot) # 35

        # Utterance encoding
        token_type_ids, attention_mask = self._make_aux_tensors(input_ids, input_len)  # input_ids = torch.Size([4, 22, 64]) input_len = torch.Size([4, 22, 2])
        # token_type_ids = torch.Size([2, 22, 64]) attention_mask = torch.Size([2, 22, 64])
        hidden, _ = self.utterance_encoder(input_ids.view(-1, self.max_seq_length), # hidden = torch.Size([44, 64, 768])
                                           token_type_ids.view(-1, self.max_seq_length),
                                           attention_mask.view(-1, self.max_seq_length),
                                           output_all_encoded_layers=False)
        hidden = torch.mul(hidden, attention_mask.view(-1, self.max_seq_length, 1).expand(hidden.size()).float())
        hidden = hidden.repeat(slot_dim, 1, 1)  # [(slot_dim*ds*ts), bert_seq, hid_size] 3 hidden = torch.Size([1540, 64, 768])

        hid_slot = self.slot_lookup.weight[target_slot, :]  # Select target slot embedding
        hid_slot = hid_slot.repeat(1, bs).view(bs * slot_dim, -1)  # [(slot_dim*ds*ts), bert_seq, hid_size]

        # Attended utterance vector
        hidden = self.attn(hid_slot, hidden, hidden,
                           mask=attention_mask.view(-1, 1, self.max_seq_length).repeat(slot_dim, 1, 1))
        hidden = hidden.squeeze()  # [slot_dim*ds*ts, bert_dim]
        hidden = hidden.view(slot_dim, ds, ts, -1).view(-1, ts, self.bert_output_dim) # hidden = torch.Size([70, 22, 768])

        # NBT
        if self.zero_init_rnn:
            h = torch.zeros(self.rnn_num_layers, input_ids.shape[0] * slot_dim, self.hidden_dim).to(
                self.device)  # [1, slot_dim*ds, hidden]
        else:
            h = hidden[:, 0, :].unsqueeze(0).repeat(self.rnn_num_layers, 1, 1)
            h = self.rnn_init_linear(h) # h = torch.Size([1, 70, 300])

        if isinstance(self.nbt, nn.GRU):
            rnn_out, _ = self.nbt(hidden, h)  # [slot_dim*ds, turn, hidden]
        elif isinstance(self.nbt, nn.LSTM):
            c = torch.zeros(self.rnn_num_layers, input_ids.shape[0] * slot_dim, self.hidden_dim).to(
                self.device)  # [1, slot_dim*ds, hidden]
            rnn_out, _ = self.nbt(hidden, (h, c))  # [slot_dim*ds, turn, hidden]
        rnn_out = self.layer_norm(self.linear(self.dropout(rnn_out))) # rnn_out = torch.Size([70, 22, 300])

        hidden = rnn_out.view(slot_dim, ds, ts, -1) # hidden = torch.Size([35, 2, 22, 768])

        # Label (slot-value) encoding
        loss = 0
        loss_slot = []
        pred_slot = []
        output = []
        for s, slot_id in enumerate(target_slot):  ## note: target_slots are successive
            # loss calculation
            hid_label = self.value_lookup[slot_id].weight
            num_slot_labels = hid_label.size(0)

            _hid_label = hid_label.unsqueeze(0).unsqueeze(0).repeat(ds, ts, 1, 1).view(ds * ts * num_slot_labels, -1)
            _hidden = hidden[s, :, :, :].unsqueeze(2).repeat(1, 1, num_slot_labels, 1).view(ds * ts * num_slot_labels,
                                                                                            -1)
            _dist = self.metric(_hid_label, _hidden).view(ds, ts, num_slot_labels)

            if self.distance_metric == "euclidean":
                _dist = -_dist
            _, pred = torch.max(_dist, -1)
            pred_slot.append(pred.view(ds, ts, 1))
            output.append(_dist)

            if labels is not None:
                _loss = self.nll(_dist.view(ds * ts, -1), labels[:, :, s].view(-1))
                loss_slot.append(_loss.item())
                loss += _loss

        if labels is None:
            return output, torch.cat(pred_slot, 2)

        # calculate joint accuracy
        pred_slot = torch.cat(pred_slot, 2)
        # print('pred slot:', pred_slot[0][0])
        # print('labels:', labels[0][0])
        accuracy = (pred_slot == labels).view(-1, slot_dim)
        acc_slot = torch.sum(accuracy, 0).float() / torch.sum(labels.view(-1, slot_dim) > -1, 0).float()
        acc = sum(torch.sum(accuracy, 1) / slot_dim).float() / torch.sum(labels[:, :, 0].view(-1) > -1, 0).float()  # joint accuracy

        
        self.acc_ = acc.item()
        self.acc_slot = acc_slot
        self.pred_slot = pred_slot
        self.loss_ = loss

        return pred_slot
        

        # if n_gpu == 1:
        #     return loss, loss_slot, acc, acc_slot, pred_slot
        # else:
        #     return loss.unsqueeze(0), None, acc.unsqueeze(0), acc_slot.unsqueeze(0), pred_slot.unsqueeze(0)
        # pass

    def loss(self, batch, loss_function):
        self.forward(batch)
        self.acc_ = 0
        return self.loss_

    def evaluate(self, batch, metric_function):
        # self.loss_ = 0
        # self.forward(batch)
        # return self.acc_, self.loss_
        self.loss_ = 0
        pred = self.predict(batch)
        _, _, labels = batch['all_input_ids'], batch['all_input_len'], batch['all_label_ids']
        metric_function.evaluate(pred, labels)

    def predict(self, batch):
        pred = self.forward(batch)
        return pred
        pass
    def initialize_slot_value_lookup(self, label_ids, slot_ids):

        self.sv_encoder.eval()

        # Slot encoding
        slot_type_ids = torch.zeros(slot_ids.size(), dtype=torch.long).to(self.device)
        slot_mask = slot_ids > 0

        hid_slot, _ = self.sv_encoder(slot_ids.view(-1, self.max_label_length),
                                      slot_type_ids.view(-1, self.max_label_length),
                                      slot_mask.view(-1, self.max_label_length),
                                      output_all_encoded_layers=False)
        # hid_slot, _ = self.sv_encoder.encode_plus(slot_ids.view(-1, self.max_label_length),
        #                                            slot_type_ids.view(-1, self.max_label_length),
        #                                slot_mask.view(-1, self.max_label_length),
        #                                output_all_encoded_layers=False)

        hid_slot = hid_slot[:, 0, :]
        hid_slot = hid_slot.detach()
        self.slot_lookup = nn.Embedding.from_pretrained(hid_slot, freeze=True)

        for s, label_id in enumerate(label_ids):
            label_type_ids = torch.zeros(label_id.size(), dtype=torch.long).to(self.device)
            label_mask = label_id > 0
            hid_label, _ = self.sv_encoder(label_id.view(-1, self.max_label_length),
                                           label_type_ids.view(-1, self.max_label_length),
                                           label_mask.view(-1, self.max_label_length),
                                           output_all_encoded_layers=False)
            hid_label = hid_label[:, 0, :]
            hid_label = hid_label.detach()
            self.value_lookup[s] = nn.Embedding.from_pretrained(hid_label, freeze=True)
            self.value_lookup[s].padding_idx = -1

        print("Complete initialization of slot and value lookup")

    def _make_aux_tensors(self, ids, len): # ids = torch.Size([4, 22, 64]); len = torch.Size([4, 22, 2])
        token_type_ids = torch.zeros(ids.size(), dtype=torch.long).to(self.device)
        for i in range(len.size(0)):
            for j in range(len.size(1)):
                if len[i, j, 0] == 0:  # padding
                    break
                elif len[i, j, 1] > 0:  # escape only text_a case
                    start = len[i, j, 0]
                    ending = len[i, j, 0] + len[i, j, 1]
                    token_type_ids[i, j, start:ending] = 1
        attention_mask = ids > 0
        return token_type_ids, attention_mask

    @staticmethod
    def init_parameter(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.GRU) or isinstance(module, nn.LSTM):
            torch.nn.init.xavier_normal_(module.weight_ih_l0)
            torch.nn.init.xavier_normal_(module.weight_hh_l0)
            torch.nn.init.constant_(module.bias_ih_l0, 0.0)
            torch.nn.init.constant_(module.bias_hh_l0, 0.0)
