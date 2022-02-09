import torch
import torch.nn as nn
import numpy as np
from numpy import array
from torch.autograd import Variable
import nltk
nltk.download('stopwords')
import pickle
import random


class BLSTM(nn.Module):
    def __init__(self, non_trainable=True):
        super(BLSTM, self).__init__()

        random.seed(1234)
        torch.manual_seed(1234)
        np.random.seed(1234)

        '''
        with open('adj_abortion/tokenized_paragraphs.pkl', 'rb') as f:
            [self.weights_matrix, self.segment2tokenized_text]=pickle.load(f)
        '''

        with open('data.pickle', 'rb') as in_file:
            [self.graph, self.id2name, self.name2id, self.stance_start, self.stance_end, self.issue_start, self.issue_end, self.ad_start, self.ad_end,\
             self.funding_entity_start, self.funding_entity_end, self.topic_unigram_start, self.topic_unigram_end, self.personal_unigram_start,\
             self.personal_unigram_end, self.policy_unigram_start, self.policy_unigram_end, self.indicator_label_start, self.indicator_label_end, \
             self.annotated_ads, self.annotated_funding_entities, self.segment2tokenized_text, self.weights_matrix, self.id2text, self.entity_indicators, \
             self.left_entity_indicators, self.right_entity_indicators, self.ad2annotated_stance, self.ad2annotated_issue,\
             self.ad2personal_policy_annotation]=pickle.load(in_file, encoding="bytes")

        self.hidden_dim = 150

        num_embeddings = len(self.weights_matrix)
        embedding_dim = len(self.weights_matrix[0])
        if torch.cuda.is_available():
            self.word_embeddings = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0).cuda()
        else:
            self.word_embeddings = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)

        self.word_embeddings.weight.data.copy_(torch.from_numpy(self.weights_matrix))

        if non_trainable:
            self.word_embeddings.weight.requires_grad = False

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if torch.cuda.is_available():
            self.lstm = nn.LSTM(300, self.hidden_dim, bidirectional=True, batch_first=True, dropout=0.5).cuda()
        else:
            self.lstm = nn.LSTM(300, self.hidden_dim, bidirectional=True, batch_first=True, dropout=0.5)

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if torch.cuda.is_available():
            return (torch.zeros(2, batch_size, self.hidden_dim).cuda(),
                    torch.zeros(2, batch_size, self.hidden_dim).cuda())
        return (torch.zeros(2, batch_size, self.hidden_dim),
                torch.zeros(2, batch_size, self.hidden_dim))

    def get_padded(self, X):
        # print X
        X_lengths = [len(sentence) for sentence in X]
        pad_token = 0
        longest_sent = max(X_lengths)
        batch_size = len(X)
        padded_X = np.ones((batch_size, longest_sent)) * pad_token

        for i, x_len in enumerate(X_lengths):
            sequence = X[i]
            padded_X[i, 0:x_len] = sequence[:x_len]
        # print padded_X
        return padded_X, X_lengths




    def forward(self, X):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.hidden = self.init_hidden(len(X))
        # batch_size, seq_len = len(X), len(X[0])
        # ---------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        X = [self.segment2tokenized_text[seg_id] for seg_id in X]
        #print X
        X, X_lengths=self.get_padded(X)
        #print X, X_lengths

        if torch.cuda.is_available():
            X = self.word_embeddings(Variable(torch.cuda.LongTensor(np.array(X))))
        else:
            X = self.word_embeddings(Variable(torch.LongTensor(np.array(X))))

        seg_sort_index = sorted(range(len(X_lengths)), key=lambda k: X_lengths[k], reverse=True)
        seg_sort_index_map = {old: new for new, old in enumerate(seg_sort_index)}
        reverse_seg_index = [seg_sort_index_map[i] for i in range(len(seg_sort_index))]
        reverse_seg_index_var = torch.LongTensor(reverse_seg_index)

        if torch.cuda.is_available():
            #X_lengths = X_lengths.cuda()
            reverse_seg_index_var = reverse_seg_index_var.cuda()

        seg_lengths_sort = sorted(X_lengths, reverse=True)
        # de-concat the document sentences in the whole batch
        #print X
        X_sort = torch.cat([X[i].unsqueeze(0) for i in seg_sort_index], 0)
        #print X_sort, seg_lengths_sort
        X = torch.nn.utils.rnn.pack_padded_sequence(X_sort, seg_lengths_sort, batch_first=True)

        # now run through LSTM
        X, self.hidden = self.lstm(X, self.hidden)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        '''
        if torch.cuda.is_available():
            embeds = torch.index_select(X, 1, torch.cuda.LongTensor(array([i for i in range(X_lengths[0])]))).sum(1) / torch.cuda.FloatTensor(array(X_lengths)).view((-1, 1))
        else:
            embeds = torch.index_select(X, 1, torch.LongTensor(array([i for i in range(X_lengths[0])]))).sum(1) / torch.FloatTensor(array(X_lengths)).view((-1, 1))

        seg_embeds = torch.index_select(embeds, 0, reverse_seg_index_var)
        return seg_embeds
        '''

        #print list(X.size()), reverse_seg_index_var

        if torch.cuda.is_available():
            seg_embeds = torch.index_select(X, 0, reverse_seg_index_var).sum(1) / torch.cuda.FloatTensor(array(X_lengths)).view((-1, 1))
        else:
            seg_embeds = torch.index_select(X, 0, reverse_seg_index_var).sum(1) / torch.FloatTensor(array(X_lengths)).view((-1, 1))

        return seg_embeds

if __name__ == '__main__':
    BLSTM=BLSTM()

    
