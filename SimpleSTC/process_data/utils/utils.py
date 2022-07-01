import random
import numpy as np
import torch
import os
from scipy.sparse import coo_matrix
from sklearn import CountVectorizer
from utils.TF_IDF import TfidfTransformer
import math
import re

def tf_idf_transform(inputs, inputs_test=None, mapping=None, sparse=False):
    vectorizer = CountVectorizer(vocabulary=mapping)
    tf_idf_transformer = TfidfTransformer()
    tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(inputs))
    weight = tf_idf.toarray()
    if inputs_test is not None:
        tf_idf_test = tf_idf_transformer.fit_transform(vectorizer.fit_transform(inputs_test), vectorizer.fit_transform(inputs))
        weight_test = tf_idf_test.toarray()
        res = np.concatenate((weight, weight_test), axis=0)
    else:
        res = weight
    if not sparse:
        res = res
    else:
        res = coo_matrix(res)
    return res

def tf_idf_out_pool(inputs, inputs_test=None, mapping=None, sparse=False):
    vectorizer = CountVectorizer(vocabulary=mapping)
    tf_idf_transformer = TfidfTransformer()
    tf_idf_test = tf_idf_transformer.fit_transform(vectorizer.fit_transform(inputs_test), vectorizer.fit_transform(inputs))
    res = tf_idf_test.toarray()
    if not sparse:
        res = res
    else:
        res = coo_matrix(res)
    return res

def PMI(inputs, mapping, window_size, sparse):
    W_ij = np.zeros([len(mapping), len(mapping)], dtype=float)
    W_i = np.zeros([len(mapping)], dtype=float)
    W_count = 0
    for one in inputs:
        word_list = one.split(' ')
        if len(word_list) - window_size < 0:
            window_num = 1
        else:
            window_num = len(word_list) - window_size + 1

        for i in range(window_num):
            W_count += 1
            context = list(set(word_list[i:i + window_size]))
            while '' in context:
                context.remove('')
            for j in range(len(context)):
                W_i[mapping[context[j]]] += 1
                for k in range(j + 1, len(context)):
                    W_ij[mapping[context[j]], mapping[context[k]]] += 1
                    W_ij[mapping[context[k]], mapping[context[j]]] += 1

    if not sparse:
        PMI_adj = np.zeros([len(mapping), len(mapping)], dtype=np.float64)
        for i in range(len(mapping)):
            for j in range(len(mapping)):
                PMI_adj[i, j] = math.log(W_ij[i, j] * W_count / W_i[i] / W_i[j]) if W_ij[i, j] != 0 else 0
                if i == j: PMI_adj[i, j] = 1

                if PMI_adj[i, j] <= 0:
                    PMI_adj[i, j] = 0

    else:
        rows = []
        columns = []
        data = []
        for i in range(len(mapping)):
            for j in range(i, len(mapping)):
                value = math.log(W_ij[i, j] * W_count / W_i[i] / W_i[j]) if W_ij[i, j] != 0 else 0
                if i == j: value = 1

                if value > 0:
                    rows.append(i)
                    columns.append(j)
                    data.append(value)
                    if i != j:
                        rows.append(j)
                        columns.append(i)
                        data.append(value)

        PMI_adj = coo_matrix((data, (rows, columns)), shape=(len(mapping), len(mapping)))
    return PMI_adj

def clean_str(string,use=True):

    if not use: return string

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_stopwords(filepath='./stopwords_en.txt'):
    stopwords = set()
    test = {}
    with open(filepath, 'r') as f:
        for line in f:
            swd = line.strip()
            stopwords.add(swd)
            if swd in test.keys():
                test[swd] += 1
            else:
                test[swd] = 0
    print(len(stopwords))
    for key in test:
        if test[key] > 0:
            print(key)
    return stopwords