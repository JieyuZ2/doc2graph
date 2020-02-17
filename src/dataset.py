import os
from collections import defaultdict
from string import punctuation as punctuation_
import nltk
from nltk.corpus import stopwords
import torch
from torchtext import data, datasets
from torchtext.vocab import Vectors, GloVe

stopwords_ = set(stopwords.words('english'))
ignore_ = list(stopwords_) + list(punctuation_)


class Dataset:
    def __init__(self, emb_dim, data_file, include_all=False):
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy')
        self.LABEL = data.Field(sequential=False, unk_token=None)
        self.MASK = data.Field(init_token='<start>', eos_token='<eos>', lower=True, tokenize='spacy')

        f = lambda ex: len(ex.text) >= 10 and len(ex.text) <= 500

        self.train, self.val, self.test, self.all = Dataset_.splits(self.TEXT, self.LABEL, self.MASK, root=data_file,
                                                                    filter_pred=f, include_all=include_all)

        if include_all:
            self.TEXT.build_vocab(self.all, vectors=GloVe('6B', dim=emb_dim))
            self.LABEL.build_vocab(self.all)
            self.MASK.build_vocab(self.all)
        else:
            self.TEXT.build_vocab(self.train)
            self.LABEL.build_vocab(self.train)
            self.MASK.build_vocab(self.train)

        self.MASK.vocab.stoi = defaultdict(lambda:1)
        for s in [self.MASK.unk_token, self.MASK.pad_token, self.MASK.init_token, self.MASK.eos_token]:
            self.MASK.vocab.stoi[s] = 0
        for s in ignore_:
            self.MASK.vocab.stoi[s] = 0

        self.n_vocab = len(self.TEXT.vocab.itos)
        self.n_labels = len(self.LABEL.vocab.itos)
        self.n_train = len(self.train)
        self.emb_dim = emb_dim

    def create_padding_mask(self, inputs, device, padding=1):
        mask = torch.ones_like(inputs).to(device)
        mask[inputs == padding] = 0
        return mask

    def create_mask(self, inputs, device, except_list=[0, 1, 2, 3], stopwords=True, punctuation=True):
        mask = torch.ones_like(inputs).to(device)
        if stopwords:
            except_list += [self.TEXT.vocab.stoi[i] for i in stopwords_]
        if punctuation:
            except_list += [self.TEXT.vocab.stoi[i] for i in punctuation_]
        for ele in except_list:
            mask[inputs == ele] = 0
        return mask

    def build_all_iter(self, batch_size=128, device=-1):
        all_iter = data.BucketIterator(
            self.all, batch_size=batch_size, device=device, shuffle=True, repeat=False
        )
        return all_iter

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

    def idx2word(self, idx):
        return self.TEXT.vocab.itos[idx]

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

    def idx2label(self, idx):
        return self.LABEL.vocab.itos[idx]

    def get_batch(self, iter_, gpu=False):
        for batch in iter_:
            if gpu:
                yield batch.text.cuda(), batch.label.cuda()
            else:
                yield batch.text, batch.label


class Dataset_(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, mask_field, data_list, **kwargs):

        fields = [('text', text_field), ('label', label_field), ('mask', mask_field)]
        examples = []

        for data_tuple in data_list:
            label, text = data_tuple[0], data_tuple[1]
            examples.append(data.Example.fromlist([text, label, text], fields))

        super(Dataset_, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, mask_field, root, split_ratio=(0.8, 0.1), include_all=False, **kwargs):
        raw_data = []

        with open(root, 'r') as f:
            for line in f:
                label, text = line.strip().split('\t ')
                label = int(label)
                raw_data.append((label, text))

        train_sample_num = round(len(raw_data)*split_ratio[0])
        val_sample_num = round(len(raw_data)*split_ratio[1]) + train_sample_num
        train_raw_data = raw_data[0:train_sample_num]
        val_raw_data = raw_data[train_sample_num:val_sample_num]
        test_raw_data = raw_data[val_sample_num:]
        train_data = cls(path=root,  text_field=text_field, label_field=label_field, mask_field=mask_field, data_list=train_raw_data, **kwargs)
        val_data = cls(path=root,  text_field=text_field, label_field=label_field, mask_field=mask_field, data_list=val_raw_data, **kwargs)
        test_data = cls(path=root, text_field=text_field, label_field=label_field, mask_field=mask_field, data_list=test_raw_data, **kwargs)

        if include_all:
            all_data = cls(path=root, text_field=text_field, label_field=label_field, mask_field=mask_field, data_list=raw_data, **kwargs)
            return (train_data, val_data, test_data, all_data)
        else:
            return (train_data, val_data, test_data, None)

    @classmethod
    def iters(cls, batch_size=32, device=0, root='.data', vectors=None, **kwargs):
        TEXT = data.Field()
        LABEL = data.Field(sequential=False)

        train, val, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)
        LABEL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device)

