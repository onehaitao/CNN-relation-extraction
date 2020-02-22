#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6


import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class WordEmbeddingLoader(object):
    """
    A loader for pre-trained word embedding
    """

    def __init__(self, config):
        self.path_word = config.embedding_path  # path of pre-trained word embedding
        self.word_dim = config.word_dim  # dimension of word embedding

    def load_embedding(self):
        word2id = dict()  # word to wordID
        word_vec = list()  # wordID to word embedding

        word2id['PAD'] = len(word2id)  # PAD character

        with open(self.path_word, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip().split()
                if len(line) != self.word_dim + 1:
                    continue
                word2id[line[0]] = len(word2id)
                word_vec.append(np.asarray(line[1:], dtype=np.float32))

        pad_emb = np.zeros([1, self.word_dim], dtype=np.float32)  # <pad> is initialize as zero
        word_vec = np.concatenate((pad_emb, word_vec), axis=0)
        word_vec = word_vec.astype(np.float32).reshape(-1, self.word_dim)
        word_vec = torch.from_numpy(word_vec)
        return word2id, word_vec


class RelationLoader(object):
    def __init__(self, config):
        self.data_dir = config.data_dir

    def __load_relation(self):
        relation_file = os.path.join(self.data_dir, 'relation2id.txt')
        rel2id = {}
        id2rel = {}
        with open(relation_file, 'r', encoding='utf-8') as fr:
            for line in fr:
                relation, id_s = line.strip().split()
                id_d = int(id_s)
                rel2id[relation] = id_d
                id2rel[id_d] = relation
        return rel2id, id2rel, len(rel2id)

    def get_relation(self):
        return self.__load_relation()


class SemEvalDateset(Dataset):
    def __init__(self, filename, rel2id, word2id, config):
        self.filename = filename
        self.rel2id = rel2id
        self.word2id = word2id
        self.max_len = config.max_len
        self.pos_dis = config.pos_dis
        self.data_dir = config.data_dir
        self.dataset, self.label = self.__load_data()

    def __get_pos_index(self, x):
        if x < -self.pos_dis:
            return 0
        if x >= -self.pos_dis and x <= self.pos_dis:
            return x + self.pos_dis + 1
        if x > self.pos_dis:
            return 2 * self.pos_dis + 2

    def __get_relative_pos(self, x, entity_pos):
        if x < entity_pos[0]:
            return self.__get_pos_index(x-entity_pos[0])
        elif x > entity_pos[1]:
            return self.__get_pos_index(x-entity_pos[1])
        else:
            return self.__get_pos_index(0)

    def __symbolize_sentence(self, e1_pos, e2_pos, sentence):
        """
            Args:
                e1_pos (tuple) span of e1
                e2_pos (tuple) span of e2
                sentence (list)
        """
        mask = [1] * len(sentence)
        if e1_pos[0] < e2_pos[0]:
            for i in range(e1_pos[0], e2_pos[1]+1):
                mask[i] = 2
            for i in range(e2_pos[1]+1, len(sentence)):
                mask[i] = 3
        else:
            for i in range(e2_pos[0], e1_pos[1]+1):
                mask[i] = 2
            for i in range(e1_pos[1]+1, len(sentence)):
                mask[i] = 3

        words = []
        pos1 = []
        pos2 = []
        length = min(self.max_len, len(sentence))
        mask = mask[:length]

        for i in range(length):
            words.append(self.word2id.get(sentence[i].lower(), self.word2id['*UNKNOWN*']))
            pos1.append(self.__get_relative_pos(i, e1_pos))
            pos2.append(self.__get_relative_pos(i, e2_pos))

        if length < self.max_len:
            for i in range(length, self.max_len):
                mask.append(0)  # 'PAD' mask is zero
                words.append(self.word2id['PAD'])

                pos1.append(self.__get_relative_pos(i, e1_pos))
                pos2.append(self.__get_relative_pos(i, e2_pos))
        unit = np.asarray([words, pos1, pos2, mask], dtype=np.int64)
        unit = np.reshape(unit, newshape=(1, 4, self.max_len))
        return unit

    def __load_data(self):
        path_data_file = os.path.join(self.data_dir, self.filename)
        data = []
        labels = []
        with open(path_data_file, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = json.loads(line.strip())
                label = line['relation']
                sentence = line['sentence']
                e1_pos = (line['subj_start'], line['subj_end'])
                e2_pos = (line['obj_start'], line['obj_end'])
                label_idx = self.rel2id[label]

                one_sentence = self.__symbolize_sentence(e1_pos, e2_pos, sentence)
                data.append(one_sentence)
                labels.append(label_idx)
        return data, labels

    def __getitem__(self, index):
        data = self.dataset[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)


class SemEvalDataLoader(object):
    def __init__(self, rel2id, word2id, config):
        self.rel2id = rel2id
        self.word2id = word2id
        self.config = config

    def __collate_fn(self, batch):
        data, label = zip(*batch)  # unzip the batch data
        data = list(data)
        label = list(label)
        data = torch.from_numpy(np.concatenate(data, axis=0))
        label = torch.from_numpy(np.asarray(label, dtype=np.int64))
        return data, label

    def __get_data(self, filename, shuffle=False):
        dataset = SemEvalDateset(filename, self.rel2id, self.word2id, self.config)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=2,
            collate_fn=self.__collate_fn
        )
        return loader

    def get_train(self):
        return self.__get_data('train.json', shuffle=True)

    def get_dev(self):
        return self.__get_data('test.json', shuffle=False)

    def get_test(self):
        return self.__get_data('test.json', shuffle=False)


if __name__ == '__main__':
    from config import Config
    config = Config()
    word2id, word_vec = WordEmbeddingLoader(config).load_embedding()
    rel2id, id2rel, class_num = RelationLoader(config).get_relation()
    loader = SemEvalDataLoader(rel2id, word2id, config)
    test_loader = loader.get_train()

    min_v, max_v = float('inf'), -float('inf')
    for step, (data, label) in enumerate(test_loader):
        # print(type(data), data.shape)
        # print(type(label), label.shape)
        # break
        pos1 = data[:, 1, :].view(-1, config.max_len)
        pos2 = data[:, 2, :].view(-1, config.max_len)
        mask = data[:, 3, :].view(-1, config.max_len)
        min_v = min(min_v, torch.min(pos1).item())
        max_v = max(max_v, torch.max(pos1).item())
        min_v = min(min_v, torch.min(pos2).item())
        max_v = max(max_v, torch.max(pos2).item())
    print(min_v, max_v)
