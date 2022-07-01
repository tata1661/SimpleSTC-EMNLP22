import numpy as np
import pickle as pkl
import json
import paddle
import paddle.nn.functional as F
import paddle.optimizer as optim
import time
from sklearn import metrics
from model.SimpleSTC import SimpleSTC
from tqdm import tqdm
from utils.utils import fetch_tensor
import os

class Trainer(object):
    def __init__(self, params):
        self.dataset = params.dataset
        self.max_epoch = params.max_epoch
        self.hidden_size = params.hidden_size
        self.device = params.device
        self.lr = params.lr
        self.weight_decay = params.weight_decay
        self.params = params
        self.data_path = params.data_path

        self.adj_dict, self.features_dict, self.train_idx, self.valid_idx, self.test_idx, self.labels, word_num = self.load_data(self.data_path)
        self.label_num = len(set(self.labels))
        self.labels = paddle.to_tensor(self.labels, place=self.device)
        word_emb_size = self.hidden_size + self.features_dict['word_emb'].shape[-1]
        self.out_features_dim = [self.hidden_size, self.hidden_size, self.hidden_size, self.label_num]
        self.in_features_dim = [word_num, word_emb_size, self.hidden_size, self.hidden_size]
        self.train_idx = paddle.to_tensor(self.train_idx, place=self.device)

        self.model = SimpleSTC(self.in_features_dim, self.out_features_dim, self.train_idx, self.params)
        self.model = self.model.to(self.device)
        total_trainable_params = sum(paddle.numel(p) for p in self.model.parameters())
        print(f'{total_trainable_params.item():,} training parameters.')
        self.optim = optim.Adam(learning_rate=self.lr, 
                                parameters = self.model.parameters(), weight_decay=self.weight_decay)
    
    def train(self):
        global_best_acc = 0
        global_best_f1 = 0
        global_best_epoch = 0
        best_test_acc = 0
        best_test_f1 = 0
        best_valid_epoch = 0
        best_valid_f1=0
        best_valid_acc = 0
        acc_valid = 0
        loss_valid = 0
        f1_valid = 0
        acc_test=0
        loss_test = 0
        f1_test = 0
        for i in (range(1, self.max_epoch + 1)):
            t=time.time()
            output = self.model(self.adj_dict, self.features_dict)
            train_scores = output
            train_labels = self.labels[self.train_idx]
            loss_train = F.cross_entropy(train_scores, train_labels)
            self.optim.clear_grad()
            acc_train = paddle.cast(paddle.equal(paddle.argmax(train_scores, axis=-1), train_labels), dtype='float64').mean().item()
            loss_train.backward()
            self.optim.step()
            loss_train = loss_train.item()
            if i%1 == 0:
                acc_valid, loss_valid, f1_valid, acc_test, loss_test, f1_test = self.test(i) 
                if acc_test > global_best_acc:
                    global_best_acc = acc_test
                    global_best_f1 = f1_test
                    global_best_epoch = i
                if acc_valid > best_valid_acc:
                    best_valid_acc = acc_valid
                    best_valid_f1 = f1_valid 
                    best_test_acc = acc_test 
                    best_test_f1 = f1_test
                    best_valid_epoch = i
                    # self.save('./model/model/')
                print('Epoch {}  loss: {:.4f} acc: {:.4f} time{:.4f}'.format(i, loss_train, acc_train, time.time()-t))
            if i%100==0:
                print('VALID: VALID ACC', best_valid_acc, ' VALID F1', best_valid_f1, 'EPOCH', best_valid_epoch) 
                print('VALID: TEST ACC', best_test_acc, 'TEST F1', best_test_f1, 'EPOCH', best_valid_epoch)
                print('GLOBAL: TEST ACC', global_best_acc, 'TEST F1', global_best_f1, 'EPOCH', global_best_epoch)
        return best_test_acc, best_test_f1
    
    def test(self, epoch):
        t = time.time()
        self.model.training = False
        output = self.model.inference(self.adj_dict, self.features_dict)
        with paddle.no_grad():
            valid_scores = output[self.valid_idx]
            valid_labels = self.labels[self.valid_idx]
            loss_valid = F.cross_entropy(valid_scores, valid_labels).item()
            acc_valid = paddle.cast(paddle.equal(paddle.argmax(valid_scores, axis=-1), valid_labels), dtype='float64').mean().item()
            f1_valid = metrics.f1_score(valid_labels.detach().cpu().numpy(), paddle.argmax(valid_scores,-1).detach().cpu().numpy(),average='macro')
            test_scores = output[self.test_idx]
            test_labels = self.labels[self.test_idx]
            loss_test = F.cross_entropy(test_scores, test_labels).item()
            acc_test = paddle.cast(paddle.equal(paddle.argmax(test_scores, axis=-1), test_labels),dtype='float64').mean().item()
            f1_test = metrics.f1_score(test_labels.detach().cpu().numpy(), paddle.argmax(test_scores,-1).detach().cpu().numpy(),average='macro')
            # print('Valid  loss: {:.4f}  acc: {:.4f}  f1: {:.4f}'.format(loss_valid, acc_valid, f1_valid),
            # 'Test  loss: {:.4f} acc: {:.4f} f1: {:.4f} time: {:.4f}'.format(loss_test, acc_test, f1_test, time.time() - t))
        self.model.training = True
        # print('test time: ', time.time() - t)
        return acc_valid, loss_valid, f1_valid, acc_test, loss_test, f1_test  

    def load_data(self, data_path):
        start=time.time()
        adj_query2word = pkl.load(open(data_path + './{}/adj_query2word.pkl'.format(self.dataset), 'rb'))
        adj_word = pkl.load(open(data_path + './pool/adj_word.pkl', 'rb'))
        word_embs = pkl.load(open(data_path + './pool/word_emb_map.pkl', 'rb'))
        word_embs = np.array(word_embs, dtype=np.float64)
        train_idx, valid_idx, test_idx = json.load(open(data_path + './{}/text_index.json'.format(self.dataset), 'r'))
        labels = json.load(open(data_path + './{}/labels.json'.format(self.dataset), 'r'))
        print('Length of [trian, valid, test, total]:', [len(train_idx), len(valid_idx), len(test_idx), len(labels)])
        adj_dict, feature_dict = {}, {}
        adj_dict['q2w'] = adj_query2word
        adj_dict['word'] = adj_word
        word_num = adj_dict['q2w'].shape[1]
        feature_dict['word'] =  np.eye(word_num, dtype=np.float64)
        feature_dict['word_emb'] = word_embs
        adj, feature = {}, {}
        for i in adj_dict.keys():
            adj[i] = fetch_tensor(adj_dict, i, self.device)
        for i in feature_dict.keys():
            feature[i] = fetch_tensor(feature_dict, i, self.device)
        print('data process time: {}'.format(time.time()-start))
        return adj, feature, train_idx, valid_idx, test_idx, labels, word_num

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        paddle.save(self.model, path + './best_model_{}.pkl'.format(self.dataset))

    def load(self, path):
        model = paddle.load(path + './best_model_{}.pkl'.format(self.dataset))
        return model


