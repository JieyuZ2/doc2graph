import os
import torch
from torchtext import data, datasets
from torchtext.vocab import Vectors, GloVe

from src.nytimes import NYnews
from src.dblp import DBLP


class Dataset:
    def __init__(self, emb_dim):
        self.TEXT.build_vocab(self.train, vectors=GloVe('6B', dim=emb_dim))
        self.LABEL.build_vocab(self.train)

        self.n_vocab = len(self.TEXT.vocab.itos)
        self.n_labels = len(self.LABEL.vocab.itos)
        self.n_train = len(self.train)
        self.emb_dim = emb_dim

    def create_padding_mask(self, inputs, device, padding=1):
        mask = torch.ones_like(inputs).to(device)
        mask[inputs == padding] = 0
        return mask

    def create_mask(self, inputs, device, except_list=[0, 1, 2, 3]):
        mask = torch.ones_like(inputs).to(device)
        for ele in except_list:
            mask[inputs == ele] = 0
        return mask

    def build_data_iter(self, batch_size=128, device=-1):
        train_iter = data.BucketIterator(
            self.train, batch_size=batch_size, device=device, shuffle=True, repeat=False
        )
        return train_iter

    def build_train_iter(self, batch_size=128, device=-1):
        train_iter = data.BucketIterator(
            self.train, batch_size=batch_size, device=device, shuffle=True, repeat=False
        )
        return train_iter

    def build_test_iter(self, batch_size=128, device=-1, shuffle=False):
        val_iter, test_iter = data.BucketIterator.splits(
            (self.val, self.test), batch_size=batch_size, device=device, shuffle=shuffle, repeat=False
        )
        return val_iter, test_iter

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def get_batch(self, iter_, gpu=False):
        for batch in iter_:
            if gpu:
                yield batch.text.cuda(), batch.label.cuda()
            else:
                yield batch.text, batch.label

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

    def idx2label(self, idx):
        return self.LABEL.vocab.itos[idx]


class NYnews_Dataset(Dataset):
    def __init__(self, emb_dim=100):
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy')
        self.LABEL = data.Field(sequential=False, unk_token=None)

        f = lambda ex: len(ex.text) >= 10 and len(ex.text) <= 500

        self.train, self.val, self.test = NYnews.splits(self.TEXT, self.LABEL, filter_pred=f)

        super(NYnews_Dataset, self).__init__(emb_dim)


class DBLP_Dataset(Dataset):
    def __init__(self, emb_dim=100):
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy')
        self.LABEL = data.Field(sequential=False, unk_token=None)

        f = lambda ex: len(ex.text) >= 10 and len(ex.text) <= 200

        self.train, self.val, self.test = DBLP.splits(self.TEXT, self.LABEL, filter_pred=f)

        super(DBLP_Dataset, self).__init__(emb_dim)


class SST_Dataset(Dataset):
    def __init__(self, emb_dim=100):
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy')
        self.LABEL = data.Field(sequential=False, unk_token=None)

        # Only take sentences with length <= 15
        # f = lambda ex: ex.label != 'neutral' and len(ex.text) >= 20
        f = lambda ex: len(ex.text) >= 10 and len(ex.text) <= 100

        self.train, self.val, self.test = datasets.SST.splits(
            self.TEXT, self.LABEL, fine_grained=False, train_subtrees=False, filter_pred=f
        )

        super(SST_Dataset, self).__init__(emb_dim)
