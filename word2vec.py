import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys


class Word2Vec:
    def __init__(self,
                 skip_data,
                 skip_model,
                 output_file_name,
                 batch_size=200,
                 window_size=5,
                 iteration=7,
                 initial_lr=10):
        """Initilize class parameters.

        Args:
            output_file_name: Name of the final embedding file.
            emb_dimention: Embedding dimention, typically from 50 to 500.
            batch_size: The count of word pairs for one forward.
            window_size: Max skip length between words.
            iteration: Control the multiple training iterations.
            initial_lr: Initial learning rate.
            min_count: The minimal word frequency, words with lower frequency will be filtered.

        Returns:
            None.
        """
        self.data = skip_data
        self.output_file_name = output_file_name
        self.emb_size = len(self.data.word2id)
        #  self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.window_size = window_size
        self.iteration = iteration
        self.initial_lr = initial_lr
        self.skip_gram_model = skip_model
        #  self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.skip_gram_model.cuda()
        self.optimizer = optim.SGD(
            self.skip_gram_model.parameters(), lr=self.initial_lr)

    def train(self):
        """Multiple training.

        Returns:
            None.
        """
        pair_count = self.data.evaluate_pair_count(self.window_size)
        #  pair_count = len(self.data.pairs)
        batch_count = self.iteration * (pair_count / self.batch_size)
        process_bar = tqdm(range(int(batch_count)))
        # self.skip_gram_model.save_embedding(
        #     self.data.id2word, 'begin_embedding.txt', self.use_cuda)
        for i in process_bar:
            pos_pairs = self.data.get_batch_pairs2(self.batch_size,
                                                  self.window_size)
            #  pos_pairs = self.data.get_batch_pairs(self.batch_size)
            neg_v = self.data.get_neg_v_neg_sampling(pos_pairs, 5)
            pos_u = [pair[0] for pair in pos_pairs]
            pos_v = [pair[1] for pair in pos_pairs]
            #  pos_u = pos_pairs[:, 0]
            #  pos_v = pos_pairs[:, 1]

            #  pos_u = Variable(torch.LongTensor(pos_u))
            #  pos_v = Variable(torch.LongTensor(pos_v))
            #  neg_v = Variable(torch.LongTensor(neg_v))
            pos_u = torch.LongTensor(pos_u)
            pos_v = torch.LongTensor(pos_v)
            neg_v = torch.LongTensor(neg_v)
            if self.use_cuda:
                pos_u = pos_u.cuda()
                pos_v = pos_v.cuda()
                neg_v = neg_v.cuda()

            self.optimizer.zero_grad()
            loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
            loss.backward()
            #  if self.skip_gram_model.lm.cha_encoder == 'bilstm':
                #  torch.nn.utils.clip_grad_norm_(self.skip_gram_model.parameters(), 0.25)
            torch.nn.utils.clip_grad_norm_(self.skip_gram_model.parameters(), 0.25)
            self.optimizer.step()

            process_bar.set_description("Loss: %0.8f, lr: %0.6f" %
                                        (loss.item(),
                                        #  (loss.data[0],
                                         self.optimizer.param_groups[0]['lr']))
            if i * self.batch_size % 100000 == 0:
                lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        self.skip_gram_model.save_word_embedding_large(self.output_file_name)


if __name__ == '__main__':
    w2v = Word2Vec(input_file_name=sys.argv[1], output_file_name=sys.argv[2])
    w2v.train()
