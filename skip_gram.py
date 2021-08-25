import os
import torch
#  from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    """Skip gram model of word2vec.

    Attributes:
        emb_size: Embedding size.
        emb_dimention: Embedding dimention, typically from 50 to 500.
        u_embedding: Embedding for center word.
        v_embedding: Embedding for neibor words.
    """

    def __init__(self, emb_size, emb_dimension, use_word=False):
        """Initialize model parameters.

        Apply for two embedding layers.
        Initialize layer weight

        Args:
            emb_size: Embedding size.
            emb_dimention: Embedding dimention, typically from 50 to 500.

        Returns:
            None
        """
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.use_word = use_word
        #  self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        #  self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        #  self.u_embeddings = nn.Embedding(emb_size, emb_dimension)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension)
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.emb_dimension
        #  self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def encode_u(self, pos_u):
        pos_u = pos_u.cuda()
        pos_u_ngram, pos_u_ngram_len = self.corpus.get_ngram_data(pos_u)
        # add 1 dim
        pos_u_ngram = pos_u_ngram.unsqueeze(0)
        pos_u_ngram_len = pos_u_ngram_len.unsqueeze(0)
        emb = self.lm.bilstm_forward(pos_u_ngram, pos_u_ngram_len)

        emb = emb.squeeze(0)
        if self.use_word:
            # following code will cause some bugs which are hard to debug
            emb_word = self.lm.encoder(pos_u)
            emb = emb + emb_word
        return emb

    def forward(self, pos_u, pos_v, neg_v):
        #  emb_u = self.u_embeddings(pos_u)
        emb_u = self.encode_u(pos_u)
        emb_v = self.v_embeddings(pos_v)
        score = torch.mul(emb_u, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        neg_emb_v = self.v_embeddings(neg_v)
        neg_score = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)
        return -1 * (torch.sum(score)+torch.sum(neg_score))

    def save_embedding(self, id2word, file_name, use_cuda):
        """Save all embeddings to file.

        As this class only record word id, so the map from id to word has to be transfered from outside.

        Args:
            id2word: map from word id to word.
            file_name: file name.
        Returns:
            None.
        """
        #  if use_cuda:
            #  embedding = self.u_embeddings.weight.cpu().data.numpy()
        #  else:
            #  embedding = self.u_embeddings.weight.data.numpy()
        #  fout = open(file_name, 'w')
        #  fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        #  embedding = self.encode_u(torch.tensor(list(range(len(id2word)))))
        #  for w_index, w in enumerate(id2word, 0):
            #  e = embedding[w_index]
            #  e = ' '.join(map(lambda x: str(x), e))
            #  fout.write('%s %s\n' % (w, e))

        os.system(f"rm -f {file_name}")
        self.eval()
        emb = self.encode_u(torch.tensor(list(range(len(id2word)))))
        print(f"SkipGramModel is current in training {self.training}")
        with open(file_name, 'x') as fh:
            fh.write(f"{emb.size(0)} {emb.size(1)}\n")
            for word, vec in zip(id2word, emb):
                str_vec = [f"{x.item():5.4f}" for x in vec]
                line = word + " " + " ".join(str_vec) + "\n"
                fh.write(line)

    def save_word_embedding_large(self, file_name):
        os.system(f"rm -f {file_name}")
        fh = open(file_name, "w")
        dict_ids = torch.tensor(list(range(len(self.corpus.dict_for_ngram.idx2word)))).cuda()
        loop_batch = 1000
        loop_batch_n = int(len(dict_ids) / loop_batch) + (len(dict_ids) % loop_batch > 0)
        fh.write(f"{len(dict_ids)} {self.emb_size}\n")
        for i in range(loop_batch_n):
            if i * loop_batch >= len(dict_ids):
                break
            loop_ids = dict_ids[i*loop_batch:(i+1)*loop_batch] 
            loop_words = self.corpus.dict_for_ngram.idx2word[i*loop_batch:(i+1)*loop_batch]
            emb = self.encode_u(loop_ids)
            for word, vec in zip(loop_words, emb):
                str_vec = [f"{x.item():5.4f}" for x in vec]
                line = word + " " + " ".join(str_vec) + "\n"
                fh.write(line)
            
        fh.close()


def test():
    model = SkipGramModel(100, 100)
    id2word = dict()
    for i in range(100):
        id2word[i] = str(i)
    model.save_embedding(id2word)


if __name__ == '__main__':
    test()
