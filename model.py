import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn.functional as F

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
        self, rnn_type, input_tokens, input_unseen_tag_index, ntoken, ninp, nhid,
        nlayers, dropout=0.5, tie_weights=False, ngram_num=None, ngram_pad_idx=0
    ):
        super(RNNModel, self).__init__()
        self.input_tokens = input_tokens
        self.input_unseen_tag_index = input_unseen_tag_index

        self.bilstm = nn.LSTM(ninp, ninp, bidirectional=True)
        # forward linear transform
        self.w_f = nn.Linear(ninp, ninp, bias=False)
        # backward linear transform
        self.w_b = nn.Linear(ninp, ninp, bias=True)

        self.drop = nn.Dropout(dropout)
        self.tie_weights = tie_weights
        self.ninp = ninp
        self.ngram_pad_idx = ngram_pad_idx
        self.ngram_encoder = nn.Embedding(ngram_num, ninp, padding_idx=self.ngram_pad_idx)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)

        self.decoder = nn.Linear(nhid, ntoken)
        self.nhid = nhid
        self.ntoken = ntoken

        if self.tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            if not self.use_word:
                raise ValueError('When using the tied flag, word input must be used')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def model_info(self, note):
        """print model's information"""
        param_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        param_num =  f"{param_num / (10**6):5.3f}"
        print(f"{note} total param_num: {param_num}M")
        print(f"{note} decoder vocab size: {self.decoder.weight.size(0)}")
        print(f"{note} ngram vocab size: {self.ngram_encoder.weight.size(0)}")

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.ngram_encoder.weight.data.uniform_(-initrange, initrange)
        # padding_idx's value is zero
        self.ngram_encoder.weight.data[self.ngram_pad_idx].zero_()
        self.w_b.bias.data.zero_()

    def bilstm_forward(self, ngram_data, ngram_length):
        seq_len, batch_size, max_ngram_len = ngram_data.size()
        # shape: max_ngram_len * (bptt*batch), first is seq_len and second is batch for bilstm
        ngram_data = ngram_data.view(-1, max_ngram_len).t()
        ngram_length = ngram_length.view(-1)
        sorted_ngram_length, perm_idx = torch.sort(ngram_length, 0, descending=True)
        sorted_ngram_data = ngram_data[:, perm_idx]
        emb_ngram = self.drop(self.ngram_encoder(sorted_ngram_data))
        packed_input = pack_padded_sequence(emb_ngram, sorted_ngram_length)
        packed_output, (ht, ct) = self.bilstm(packed_input)
        # bilstm_output's shape (sorted): max_ngram_len * (bptt*batch) * (dim*2)
        bilstm_output, _ = pad_packed_sequence(packed_output)
        # forward final output + backward final output
        # ht[0, :, :]'shape is (bptt*batch)*dim, 0 to index forward part
        transformed_bilstm_output = self.w_f(ht[0, :, :]) + self.w_b(ht[1, :, :])
        # unsort (ascending)
        _, unsort_indexs = torch.sort(perm_idx, 0)
        # shape: (bptt * batch) * dim
        transformed_bilstm_output = transformed_bilstm_output[unsort_indexs]
        # shape: bptt * batch * dim
        emb_ngram = transformed_bilstm_output.view(seq_len, batch_size, self.ninp)
        return emb_ngram

    def get_ngram_dict_emb(self, ngram_data, ngram_data_len):
        # add 1 dim
        ngram_data = ngram_data.unsqueeze(0)
        ngram_data_len = ngram_data_len.unsqueeze(0)
        emb = self.bilstm_forward(ngram_data, ngram_data_len)
        emb = emb.squeeze(0)
        return emb

    def forward(self, input, ngram_data, ngram_length, hidden):
        emb_ngram = self.bilstm_forward(ngram_data, ngram_length)
        output, hidden = self.rnn(emb_ngram, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
