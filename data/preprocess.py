#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import json
import re
from nltk.tokenize import word_tokenize


def search_entity(sentence):
    e1 = re.findall(r'<e1>(.*)</e1>', sentence)[0]
    e2 = re.findall(r'<e2>(.*)</e2>', sentence)[0]
    sentence = sentence.replace('<e1>' + e1 + '</e1>', ' <e1> ' + e1 + ' </e1> ', 1)
    sentence = sentence.replace('<e2>' + e2 + '</e2>', ' <e2> ' + e2 + ' </e2> ', 1)
    sentence = word_tokenize(sentence)
    sentence = ' '.join(sentence)
    sentence = sentence.replace('< e1 >', '<e1>')
    sentence = sentence.replace('< e2 >', '<e2>')
    sentence = sentence.replace('< /e1 >', '</e1>')
    sentence = sentence.replace('< /e2 >', '</e2>')
    sentence = sentence.split()

    assert '<e1>' in sentence
    assert '<e2>' in sentence
    assert '</e1>' in sentence
    assert '</e2>' in sentence

    subj_start = subj_end = obj_start = obj_end = 0
    pure_sentence = []
    for i, word in enumerate(sentence):
        if '<e1>' == word:
            subj_start = len(pure_sentence)
            continue
        if '</e1>' == word:
            subj_end = len(pure_sentence) - 1
            continue
        if '<e2>' == word:
            obj_start = len(pure_sentence)
            continue
        if '</e2>' == word:
            obj_end = len(pure_sentence) - 1
            continue
        pure_sentence.append(word)
    return e1, e2, subj_start, subj_end, obj_start, obj_end, pure_sentence


def convert(path_src, path_des):
    with open(path_src, 'r', encoding='utf-8') as fr:
        data = fr.readlines()
    with open(path_des, 'w', encoding='utf-8') as fw:
        for i in range(0, len(data), 4):
            id_s, sentence = data[i].strip().split('\t')
            sentence = sentence[1:-1]
            e1, e2, subj_start, subj_end, obj_start, obj_end, sentence = search_entity(sentence)
            meta = dict(
                id=id_s,
                relation=data[i+1].strip(),
                head=e1,
                tail=e2,
                subj_start=subj_start,
                subj_end=subj_end,
                obj_start=obj_start,
                obj_end=obj_end,
                sentence=sentence,
                comment=data[i+2].strip()[8:]
            )
            json.dump(meta, fw, ensure_ascii=False)
            fw.write('\n')


if __name__ == '__main__':
    path_train = './SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
    path_test = './SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'

    convert(path_train, 'train.json')
    convert(path_test, 'test.json')
