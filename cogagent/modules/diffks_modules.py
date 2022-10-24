# Coding: utf-8
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.modules.rnn import GRU
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


import torch
import os


_VF = torch._C._VariableFunctions
F_GRUCell = _VF.gru_cell


def sortSequence(data, length):
    shape = data.shape
    len, fsize = shape[0], shape[-1]
    data = data.reshape(len, -1, fsize)
    batch_size = data.shape[1]
    length = length.reshape(-1)

    zero_num = np.sum(length == 0)
    memo = list(reversed(np.argsort(length).tolist()))[:batch_size - zero_num]
    res = torch.zeros([data.shape[0], batch_size - zero_num, data.shape[-1]],dtype=data.dtype,device=data.device)
    for i, idx in enumerate(memo):
        res[:, i, :] = data[:, idx, :]
    return res, sorted(length, reverse=True)[: batch_size - zero_num], (shape, memo, zero_num)


def sortSequenceByMemo(data, memo):
    data = data.reshape(-1, data.shape[-1])
    batch_size = data.shape[0]
    shape, memo, zero_num = memo
    res = torch.zeros([batch_size - zero_num, data.shape[-1]], dtype=data.dtype, device=data.device)
    for i, idx in enumerate(memo):
        res[i, :] = data[idx, :]
    return res


def revertSequence(data, memo, isseq=False):
    shape, memo, zero_num = memo
    if isseq:
        res = torch.zeros([data.shape[0], data.shape[1] + zero_num, data.shape[2]],dtype=data.dtype,device=data.device)
        for i, idx in enumerate(memo):
            res[:, idx, :] = data[:, i, :]
        return res.reshape(*((res.shape[0],) + shape[1:-1] + (res.shape[-1],)))
    else:
        res = torch.zeros([data.shape[0] + zero_num, data.shape[1]], dtype=data.dtype,device=data.device)
        for i, idx in enumerate(memo):
            res[idx, :] = data[i, :]
        return res.reshape(*(shape[1:-1] + (res.shape[-1],)))


def flattenSequence(data, length):
    arr = []
    for i in range(length.size):
        arr.append(data[0:length[i], i])
    return torch.cat(arr, dim=0)


def copySequence(data, length):  # for BOW loss
    arr = []
    for i in range(length.size):
        arr.append(data[i].repeat(length[i], 1))
    return torch.cat(arr, dim=0)


def maskedSoftmax(data, length):
    mask = torch.tensor((np.expand_dims(np.arange(data.shape[0]), 1) < np.expand_dims(length, 0)).astype(int),dtype=data.dtype,device=data.device)
    return data.masked_fill(mask == 0, -1e9).softmax(dim=0)


class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, layers=1, bidirectional=False, initpara=True, attn_decode=False,
                 post_size=None):
        super(MyGRU, self).__init__()

        self.input_size, self.hidden_size, self.layers, self.bidirectional = \
            input_size, hidden_size, layers, bidirectional
        self.GRU = GRU(input_size, hidden_size, layers, bidirectional=bidirectional)
        self.initpara = initpara
        if initpara:
            if bidirectional:
                self.h_init = Parameter(torch.Tensor(2 * layers, 1, hidden_size))
            else:
                self.h_init = Parameter(torch.Tensor(layers, 1, hidden_size))
        self.reset_parameters()

        if attn_decode:
            self.attn_query = nn.Linear(hidden_size, post_size)

    def reset_parameters(self):
        if self.initpara:
            stdv = 1.0 / math.sqrt(self.hidden_size)
            self.h_init.data.uniform_(-stdv, stdv)

    def getInitialParameter(self, batch_size):
        return self.h_init.repeat(1, batch_size, 1)

    def forward(self, incoming, length, h_init=None, need_h=False, attn_decode=False, post=None, post_length=None):
        if not attn_decode:
            sen_sorted, length_sorted, memo = sortSequence(incoming, length)
            left_batch_size = sen_sorted.shape[-2]
            sen_packed = pack_padded_sequence(sen_sorted, length_sorted)
            if h_init is None:
                h_init = self.getInitialParameter(left_batch_size)
            else:
                h_shape = h_init.size()
                h_init = sortSequenceByMemo(h_init, memo)
                h_init = h_init.reshape(h_shape)
                if h_init.dim() < 3:
                    h_init = torch.unsqueeze(h_init, 0)

            h, h_n = self.GRU(sen_packed, h_init)
            h_n = h_n.transpose(0, 1).reshape(left_batch_size, -1)
            h_n = revertSequence(h_n, memo)
            if need_h:
                h = pad_packed_sequence(h)[0]
                h = revertSequence(h, memo, True)
                return h, h_n
            else:
                return h_n

        else:
            batch_size = incoming.shape[1]
            seqlen = incoming.shape[0]
            if h_init is None:
                h_init = self.getInitialParameter(batch_size)
            else:
                h_init = torch.unsqueeze(h_init, 0)
            h_now = h_init[0]
            hs = []
            attn_weights = []

            for i in range(seqlen):
                query = self.attn_query(h_now)
                attn_weight = maskedSoftmax((query.unsqueeze(0) * post).sum(-1), post_length)
                context = (attn_weight.unsqueeze(-1) * post).sum(0)
                h_now = self.cell_forward(torch.cat([incoming[i], context], dim=-1), h_now) * Tensor(
                    (length > np.ones(batch_size) * i).astype(float)).unsqueeze(-1)

                hs.append(h_now)
                attn_weights.append(attn_weight)

            return torch.stack(hs), h_now

    def cell_forward(self, incoming, h):
        return F_GRUCell(
            incoming, h,
            self.GRU.weight_ih_l0, self.GRU.weight_hh_l0,
            self.GRU.bias_ih_l0, self.GRU.bias_hh_l0
        )