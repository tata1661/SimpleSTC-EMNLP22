import torch
import os
import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.GCN import GCN


class SimpleSTC(nn.Module):
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
        self.complinears = nn.Linear(self.in_features_dim[1], self.out_features_dim[1], bias=False)
        self.final_GCN = GCN(self.in_features_dim[2], self.out_features_dim[2])
        self.final_GCN_2 = GCN(self.out_features_dim[2], self.out_features_dim[2])
        self.FC = nn.Linear(in_features_dim[3], self.label_num)


    def forward(self, adj, feature):
        word_embedding = self.GCNs_2(adj['word'],
                torch.relu(self.GCNs(adj['word'],feature['word'], identity=True)))
        word_embedding = torch.cat([
                F.dropout(word_embedding, p=self.drop_out, training=self.training), feature['word_emb']], dim=-1)
        refined_text_input = torch.matmul(adj['q2w'], word_embedding)/(torch.sum(adj['q2w'], dim=-1).unsqueeze(-1) + 1e-9)
    
        refined_text_input = refined_text_input[self.train_idx]  # not uni-norm
        Doc_features = self.complinears(refined_text_input)

        DocFea4ADJ = Doc_features / (Doc_features.norm(p=2, dim=-1, keepdim=True) + 1e-9) 
        refined_Doc_features = F.dropout(Doc_features, p=self.drop_out, training=self.training) 
        
        cos_simi_total = torch.matmul(DocFea4ADJ, DocFea4ADJ.t())
        refined_adj_tmp = cos_simi_total * (cos_simi_total > self.threshold).float()
        refined_Doc_adj = refined_adj_tmp / (refined_adj_tmp.sum(dim=-1, keepdim=True) + 1e-9)
        final_text_output = self.final_GCN_2(refined_Doc_adj, self.final_GCN(refined_Doc_adj, refined_Doc_features))
        final_text_output=F.dropout(final_text_output, p=self.drop_out, training=self.training)
        scores = self.FC(final_text_output)
        return scores

    def inference(self, adj, feature):
        word_embedding = self.GCNs_2(adj['word'],
                torch.relu(self.GCNs(adj['word'],feature['word'], identity=True)))
        word_embedding = torch.cat([
            F.dropout(word_embedding, p=self.drop_out, training=self.training), feature['word_emb']], dim=-1)
        refined_text_input = torch.matmul(adj['q2w'], word_embedding)/(torch.sum(adj['q2w'], dim=-1).unsqueeze(-1) + 1e-9)

        Doc_features = self.complinears(refined_text_input)
        DocFea4ADJ = Doc_features / (Doc_features.norm(p=2, dim=-1, keepdim=True) + 1e-9) 
        refined_Doc_features = F.dropout(Doc_features, p=self.drop_out, training=self.training) 
        cos_simi_total = torch.matmul(DocFea4ADJ, DocFea4ADJ[self.train_idx].t())
        refined_adj_tmp = cos_simi_total * (cos_simi_total > self.threshold).float()
        
        len_train = len(self.train_idx)
        supp_adj = (DocFea4ADJ * DocFea4ADJ).sum(dim=-1, keepdim=True)
        supp_adj[self.train_idx] = 0
        refined_adj_tmp = torch.cat([refined_adj_tmp, supp_adj], dim=-1)
        refined_Doc_adj = refined_adj_tmp / (refined_adj_tmp.sum(dim=-1, keepdim=True) + 1e-9)
        refined_Doc_adj, alpha_list = refined_Doc_adj.split([len_train, 1], dim=-1)
        Doc_train_adj = refined_Doc_adj[self.train_idx]
        Emb_train = self.final_GCN_2(Doc_train_adj, self.final_GCN(Doc_train_adj, refined_Doc_features[self.train_idx]))
        Doc_output = torch.matmul(refined_Doc_adj, Emb_train)
        emb_Doc_Feat = self.final_GCN_2.inference(self.final_GCN.inference(refined_Doc_features))
        final_text_output = Doc_output + alpha_list * emb_Doc_Feat
        final_text_output=F.dropout(final_text_output, p=self.drop_out, training=self.training)
        scores = self.FC(final_text_output)
        return scores
