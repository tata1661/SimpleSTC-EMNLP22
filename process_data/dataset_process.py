


import json
import pickle as pkl
from tqdm import tqdm
import os
import numpy as np
from utils.utils import clean_str, load_stopwords, tf_idf_out_pool


def process_raw_code(dataset):
    stop_word=load_stopwords()
    stop_word.add('')
    text_path = './corpus/{}.txt'.format(dataset)
    raw_texts = []
    with open(text_path, 'r') as f:
        for line in f.readlines():
            raw_texts.append(line.strip())
    print(len(raw_texts))
    words_list = []
    query_dict = {}
    for text in tqdm(raw_texts):
        text_info_tmp = []
        query = clean_str(text)
        if not query or query == '':
            print(query)
            continue
        text_info_tmp.append(query)
        words = [one.lower() for one in query.split(' ') if one not in stop_word]
        text_info_tmp.append(words)
        for word in words:
            if word not in words_list:
                words_list.append(word)
        if text_info_tmp in query_dict.values():
            print('repetition: {}'.format(query))
            continue
        query_dict[len(query_dict)] = text_info_tmp
    print(len(query_dict))
    os.makedirs('./data/{}'.format(dataset), exist_ok=True)
    json.dump(query_dict, open('./data/{}/{}_query_dict.json'.format(dataset, dataset), 'w'), ensure_ascii=False)

def process(dataset, pool_path):
    split_path = './split/{}.txt'.format(dataset)
    train_label_map, valid_label_map, test_label_map = {}, {}, {} #train label index
    label_list = []
    with open(split_path, 'r') as f:
        for line in f.readlines():
            temp = line.strip().split("\t")
            if temp[2] not in label_list:
                label_list.append(temp[2])
            if temp[1].find('test') != -1:
                test_label_map[int(temp[0])] = temp[2]
            elif temp[1].find('train') != -1:
                train_label_map[int(temp[0])] = temp[2]
            elif temp[1].find('valid') != -1:
                valid_label_map[int(temp[0])] = temp[2]
    label_map = {value: i for i, value in enumerate(label_list)}
    print(len(train_label_map), len(valid_label_map), len(test_label_map))
    query_dict = json.load(open('./data/{}/{}_query_dict.json'.format(dataset, dataset), 'r'))
    labels = []
    train_idx, valid_idx, test_idx= [], [], []
    word_list = []
    word_mapping = json.load(open(pool_path + './word_mapping.json', 'r'))
    print(len(word_mapping))
    for key, val in train_label_map.items():
        text_tmp = query_dict[str(key)]
        labels.append(val)
        train_idx.append(len(train_idx))
        WORD = text_tmp[1]
        word_list.append(' '.join(WORD))

    for key, val in valid_label_map.items():
        text_tmp = query_dict[str(key)]
        labels.append(val)
        valid_idx.append(len(valid_idx) + len(train_idx))
        WORD = text_tmp[1]
        word_list.append(' '.join(WORD))

    for key, val in test_label_map.items():
        text_tmp = query_dict[str(key)]
        labels.append(val)
        test_idx.append(len(test_idx) + len(valid_idx) + len(train_idx))
        WORD = text_tmp[1]
        word_list.append(' '.join(WORD))
    data_path = '../data/{}/'.format(dataset)
    os.makedirs(data_path, exist_ok=True)
    
    labels = [label_map[label] for label in labels]
    print('Length of [trian, valid, test, total]:', [len(train_idx), len(valid_idx), len(test_idx), len(labels)])
    pool_doc_list = pkl.load(open(pool_path + './word_list.pkl', 'rb'))
    adj_query2word = tf_idf_out_pool(pool_doc_list, word_list, word_mapping, sparse=False)
    word_embs = pkl.load(open(pool_path + './word_emb_map.pkl', 'rb'))
    word_embs = np.array(word_embs, dtype=np.float64)
    pkl.dump(adj_query2word, open(data_path+'./adj_query2word.pkl', 'wb'))
    json.dump([train_idx, valid_idx, test_idx], open(data_path+'./text_index.json', 'w'))
    json.dump(labels, open(data_path+'./labels.json', 'w'))

datasets = ['twitter', 'mr', 'snippets', 'tagmynews']
for data in datasets:
    process_raw_code(data)
    process(data, '../data/pool')