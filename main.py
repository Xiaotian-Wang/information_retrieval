# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import csv
import string

import jieba
import numpy as np
from rank_bm25a import BM25Okapi
from zhon import hanzi

DATA_DIR = 'data/HKBK_CT.csv'
DICT_DIR = 'data/cidian.txt'
SAVE_DIR = 'data/'
DATA_SEP_DIR = 'data/separated_data.npy'
DATA_NAME_ORIGINAL_DIR = 'data/names_original.npy'
DATA_CT_ZB_SEP_DIR = 'data/citiao_zhibiao_sep.npy'
DATA_CT_ZB_ORIGINAL_DIR = 'data/original_citiao_zhibiao.npy'

DATA_TYC_PATH = 'data/HKBK_HB_GXB_YXC_TYC.csv'
DATA_STOPWORDS_PATH = 'data/stopwords.npy'


jieba.load_userdict(DICT_DIR)


def read_data(data_dir=DATA_DIR):
    with open(data_dir, 'r') as f:
        read = csv.reader(f)
        names = []
        contents = []
        for row in read:
            names.append(row[4])
            contents.append(row[15])
        names = names[1:]
        contents = contents[1:]
        return names, contents


def word_separate(data, remove_space=True, remove_punctuation=True, for_search=False, remove_stopwords=True):
    i=0
    data_list = list()
    if remove_stopwords:
        stopwords = np.load(DATA_STOPWORDS_PATH)
        def is_in(list1, list2):
            res = list()
            for item in list1:
                if item in list2:
                    res += [True]
                else:
                    res += [False]
            return res

    for line in data:
        if for_search:
            temp = list(jieba.cut_for_search(line))
        else:
            temp = list(jieba.cut(line))
        temp = np.array(temp)
        if remove_space:
            index = np.where(temp == ' ')
            temp = np.delete(temp, index)
        if remove_punctuation:
            for item in hanzi.punctuation:
                index = np.where(temp == item)
                temp = np.delete(temp, index)
            for item in string.punctuation:
                index = np.where(temp == item)
                temp = np.delete(temp, index)
        if remove_stopwords:
            index = np.where(is_in(temp, stopwords))
            temp = np.delete(temp, index)

        #        temp = temp.tolist()
        data_list.append(temp)
        i = i+1
        if i%100 == 0:
            print(i)
    return data_list


def rank_prep(data_dir=DATA_SEP_DIR):
    data = np.load(data_dir, allow_pickle=True)

    a = list()
    for sent in data:
        a.append(' '.join(sent))
    corpus = a
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def rank_prep2(data_dir=DATA_SEP_DIR):
    data = np.load(data_dir, allow_pickle=True)
    tokenized_corpus = data.tolist()
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def load_syn(data_path=DATA_TYC_PATH):
    with open(data_path, 'r') as f:
        read = csv.reader(f)
        ysc = []
        tyc = []
        for row in read:
            ysc.append(row[2])
            tyc.append(row[3])
        ysc = ysc[1:]
        tyc = tyc[1:]
    return ysc, tyc


def add_syn(query: list, ysc: list, tyc: list):
    temp = []
    query = query.tolist()
    for item in query:
        temp += np.array(ysc)[np.where(np.array(tyc) == item)].tolist()

    return query + temp


bm25 = rank_prep()
bm25_ZB = rank_prep(data_dir=DATA_CT_ZB_SEP_DIR)


def rank_it(query_original: str, k=None, ysc=None, tyc=None, method=bm25):
    if ysc is None:
        ysc = []
    if tyc is None:
        tyc = []
    query = word_separate([query_original], for_search=True)
    query = query[0]
    query = add_syn(query, ysc=ysc, tyc=tyc)
    query = ' '.join(query)
    # ----------------------
    tokenized_query = query.split(" ")
    doc_scores = method.get_scores(tokenized_query)
    num = sum(doc_scores > 0)
    top_index = doc_scores.argsort()[::-1][0:num]
    if k is not None:
        if k < num:
            top_index = top_index[0:k]

    names = np.load('data/names_original.npy')
    print(query_original)
    print('-' * 20)
    for item in names[top_index]:
        print(item)
    return list(names[top_index])


def rank_it2(query_original: str, k=None, ysc=None, tyc=None, method=bm25_ZB):
    if ysc is None:
        ysc = []
    if tyc is None:
        tyc = []
    query = word_separate([query_original], for_search=True)
    query = query[0]
    query = add_syn(query, ysc=ysc, tyc=tyc)
    query = ' '.join(query)
    # ----------------------
    tokenized_query = query.split(" ")
    doc_scores = method.get_scores(tokenized_query)
    num = sum(doc_scores > 0)
    top_index = doc_scores.argsort()[::-1][0:num]
    if k is not None:
        if k < num:
            top_index = top_index[0:k]

    names = np.load(DATA_CT_ZB_ORIGINAL_DIR)
    print(query_original)
    print('-' * 20)
    for item in names[top_index]:
        print(item)
    return list(names[top_index])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    '''
    names, contents = read_data()
    res1 = word_separate(names, for_search=True)
    res2 = word_separate(contents)
    res = []
    res = []
    for i in range(len(res1)):
        res.append(np.append(res1[i],res2[i]))
    save_it = np.array(res)
    np.save('data/separated_data2.npy', save_it)
    '''

    '''
    print(" ")
    with open('data/zhibiao.txt', 'r') as f:
        read = f.readlines()
        zhibiao = []
        for row in read:
            row = str(row).replace('\n', '')
            zhibiao.append(row)
        zhibiao = zhibiao[1:]
    '''
    print('Hello World!')
