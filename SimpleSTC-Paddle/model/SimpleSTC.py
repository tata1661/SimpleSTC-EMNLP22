import os
import tqdm
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from model.GCN import GCN
paddle.set_default_dtype('float64')
import time


class SimpleSTC(nn.Layer):
    def __init__(self, in_features_dim, out_features_dim, train_idx, params):
        super(SimpleSTC, self).__init__()
        self.threshold = params.threshold
        self.in_features_dim = in_features_dim
        self.out_features_dim = out_features_dim
        self.label_num = out_features_dim[-1]
        self.drop_out = params.drop_out
        self.train_idx = train_idx
        
        self.GCNs = GCN(self.in_features_dim[0], self.out_features_dim[0])
        self.GCNs_2=GCN(self.out_features_dim[0], self.out_features_dim[0])
        self.complinears = nn.Linear(self.in_features_dim[1], self.out_features_dim[1])
        self.final_GCN = GCN(self.in_features_dim[2], self.out_features_dim[2])
        self.final_GCN_2 = GCN(self.out_features_dim[2], self.out_features_dim[2])
        self.FC = nn.Linear(in_features_dim[3], self.label_num)


    def forward(self, adj, feature):
        t = time.time()
        word_embedding = self.GCNs_2(adj['word'], F.relu(self.GCNs(adj['word'],feature['word'], identity=True)))
        word_embedding = paddle.concat([
                F.dropout(word_embedding, p=self.drop_out, training=self.training), feature['word_emb']], axis=-1)
        refined_text_input = paddle.matmul(adj['q2w'], word_embedding, )/(paddle.sum(adj['q2w'], axis=1, keepdim=True) + 1e-9)
        Doc_features = self.complinears(refined_text_input[self.train_idx])
        DocFea4ADJ = Doc_features / (paddle.linalg.norm(Doc_features, p=2, axis=-1, keepdim=True) + 1e-9)
        refined_Doc_features = F.dropout(Doc_features, p=self.drop_out, training=self.training) 
        cos_simi_total = paddle.matmul(DocFea4ADJ, DocFea4ADJ, transpose_y=True)
        refined_adj_tmp = cos_simi_total * (paddle.cast(cos_simi_total > self.threshold, dtype='float64'))
        refined_Doc_adj = refined_adj_tmp / (paddle.sum(refined_adj_tmp, axis=-1, keepdim=True) + 1e-9)
        final_text_output = self.final_GCN_2(refined_Doc_adj, self.final_GCN(refined_Doc_adj, refined_Doc_features))
        final_text_output=F.dropout(final_text_output, p=self.drop_out, training=self.training)
        scores = self.FC(final_text_output)
        return scores

    def inference(self, adj, feature):
        t = time.time()
        word_embedding = self.GCNs_2(adj['word'], F.relu(self.GCNs(adj['word'],feature['word'], identity=True)))
        word_embedding = paddle.concat([
            F.dropout(word_embedding, p=self.drop_out, training=self.training), feature['word_emb']], axis=-1)
        refined_text_input = paddle.matmul(adj['q2w'], word_embedding)/(paddle.sum(adj['q2w'], axis=1, keepdim=True)+ 1e-9)
        Doc_features = self.complinears(refined_text_input)
        DocFea4ADJ = Doc_features / (paddle.linalg.norm(Doc_features, p=2, axis=-1, keepdim=True) + 1e-9)
        refined_Doc_features = F.dropout(Doc_features, p=self.drop_out, training=self.training) 
        cos_simi_total = paddle.matmul(DocFea4ADJ, DocFea4ADJ[self.train_idx], transpose_y=True)
        refined_adj_tmp = cos_simi_total * (paddle.cast(cos_simi_total > self.threshold, dtype='float64'))
        
        len_train = len(self.train_idx)
        supp_adj = paddle.sum((DocFea4ADJ * DocFea4ADJ), axis=-1, keepdim=True)
        supp_adj[self.train_idx] = 0
        refined_adj_tmp = paddle.concat([refined_adj_tmp, supp_adj], axis=-1)
        refined_Doc_adj = refined_adj_tmp / (paddle.sum(refined_adj_tmp, axis=-1, keepdim=True) + 1e-9)
        refined_Doc_adj, alpha_list = refined_Doc_adj.split([len_train, 1], axis=-1)
        Doc_train_adj = refined_Doc_adj[self.train_idx]
        Emb_train = self.final_GCN_2(Doc_train_adj, self.final_GCN(Doc_train_adj, refined_Doc_features[self.train_idx]))
        Doc_output = paddle.matmul(refined_Doc_adj, Emb_train)
        emb_Doc_Feat = self.final_GCN_2.inference(self.final_GCN.inference(refined_Doc_features))
        final_text_output = Doc_output + alpha_list * emb_Doc_Feat
        final_text_output=F.dropout(final_text_output, p=self.drop_out, training=self.training)
        scores = self.FC(final_text_output)
        return scores
