
import nltk
import re
from tqdm import tqdm
import pickle as pkl
import json
import time
import numpy as np
from utils.utils import clean_str, load_stopwords, PMI

def process_raw_data(doc_list_cleaned, pool_path):
    from collections import defaultdict
    word_freq=defaultdict(int)
    word_list= []
    for item in tqdm(doc_list_cleaned):
        words = item.split(' ')
        word_list.append(' '.join(words))
        for one in words:
            word_freq[one] += 1
    pkl.dump(word_freq, open('./word_freq.pkl', 'wb'))

def process_pool(doc_list_cleaned, pool_path):
    stop_word=load_stopwords()
    stop_word.add('')
    glove_emb = pkl.load(open('./old_glove_6B/embedding_glove.p', 'rb'))
    vocab = pkl.load(open('./old_glove_6B/vocab.pkl', 'rb'))
    word_list = []
    word_mapping = {}
    word_freq = pkl.load(open('./word_freq.pkl', 'rb'))
    freq_stop=0
    maxfreq = 0
    maxword = 0
    for word,count in word_freq.items():
        if count > maxfreq:
            maxfreq = count
            maxword = word
        if count < 10:
            stop_word.add(word)
            freq_stop+=1
    print('freq_stop word num',freq_stop, 'word sum num', len(word_freq))
    print(len(stop_word), maxfreq, maxword)
    swcount = 0
    notvocab = 0
    for word in word_freq.keys():
        if word in stop_word:
            swcount += 1
        elif word not in stop_word and word not in vocab:
            notvocab += 1
        else:
            if word not in word_mapping:
                word_mapping[word] = len(word_mapping)

    print('pool num word', len(word_mapping))
    word_embs = []
    for word in word_mapping:
        if word in vocab:
            word_embs.append(glove_emb[vocab[word]])
        else:
            word_embs.append(np.zeros(300, dtype=np.float64))
    word_embs = np.array(word_embs, dtype=np.float64)
    word_list = []
    for item in tqdm(doc_list_cleaned):
        doc = item.split(' ')
        words = [one for one in doc if one in word_mapping]
        word_list.append(' '.join(words))
    pkl.dump(word_list, open(pool_path + './word_list.pkl','wb'))
    json.dump(word_mapping, open(pool_path + './word_mapping.json', 'w'))
    pkl.dump(word_embs, open(pool_path + './word_emb_map.pkl', 'wb'))

def process_pmi(pool_path):
    import time 
    t = time.time()
    word_list = pkl.load(open(pool_path + './word_list.pkl', 'rb'))
    word_mapping = json.load(open(pool_path + './word_mapping.json', 'r'))
    adj_word = PMI(word_list, word_mapping, window_size=5, sparse=True)
    adj_word = adj_word.toarray()
    print(adj_word.shape)
    pkl.dump(adj_word, open(pool_path + './adj_word.pkl', 'wb'))
    print(time.time() - t)


pool_path = './pool/'
import os
os.makedirs(pool_path, exist_ok=True)
doc_list_cleaned = []
with open('doc_list.txt', 'r') as fin:
    for line in fin.readlines():
        tmp = clean_str(line)
        doc_list_cleaned.append(tmp)
with open('doc_list_cleaned.txt', 'r') as fin:
    doc_list_cleaned = fin.readlines()
process_raw_data(doc_list_cleaned, pool_path)
process_pool(doc_list_cleaned, pool_path)
process_pmi(pool_path)

