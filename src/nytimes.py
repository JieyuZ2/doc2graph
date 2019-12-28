import os
import glob
import io
import random

import torchtext.data as data
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))


class NYnews(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, data_list, **kwargs):

        fields = [('text', text_field), ('label', label_field)]
        examples = []

        for data_tuple in data_list:
            label, text = data_tuple[0], data_tuple[1]
            examples.append(data.Example.fromlist([text, label], fields))

        super(NYnews, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, root=parent_path+'/data/processed_news.txt', split_ratio=(0.8, 0.1), **kwargs):
        raw_data = []

        with open(root, 'r') as f:
                for line in f:
                    label, text = line.strip().split('\t ')
                    label = int(label)
                    raw_data.append((label, text))

        random.shuffle(raw_data)

        train_sample_num = round(len(raw_data)*split_ratio[0])
        val_sample_num = round(len(raw_data)*split_ratio[1]) + train_sample_num
        train_raw_data = raw_data[0:train_sample_num]
        val_raw_data = raw_data[train_sample_num:val_sample_num]
        test_raw_data = raw_data[val_sample_num:]
        train_data = cls(path=root,  text_field=text_field, label_field=label_field, data_list=train_raw_data, **kwargs)
        val_data = cls(path=root,  text_field=text_field, label_field=label_field, data_list=val_raw_data, **kwargs)
        test_data = cls(path=root, text_field=text_field, label_field=label_field, data_list=test_raw_data, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

    @classmethod
    def iters(cls, batch_size=32, device=0, root='.data', vectors=None, **kwargs):
        TEXT = data.Field()
        LABEL = data.Field(sequential=False)

        train, val, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)
        LABEL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device)
