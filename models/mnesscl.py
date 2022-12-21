import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from evaluate import evaluate
from embedder import embedder
from layers import GCN, Structure_mlp, Semantic_mlp

class MNESSCL(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.k = args.k
        self.t = args.t
        self.coef_n = self.args.coef_n
        self.coef_n_v = self.args.coef_n_v
        self.coef_c = self.args.coef_c
        self.coef_c_v = self.args.coef_c_v

        if not os.path.exists(self.args.save_folder):
            os.makedirs(self.args.save_folder)

    def training(self):
        features = self.features.to(self.args.device)
        features_pos = self.features_pos.to(self.args.device)
        adj_list = [adj.to(self.args.device) for adj in self.adj_list]
        adj_pos_list = [adj.to(self.args.device) for adj in self.adj_pos_list]

        print("Started training...")
        model = modeler(self.args.ft_size, self.args.hid_units, len(adj_list), self.k, self.t).to(self.args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        cnt_wait = 0
        best = 1e9
        model.train()
        for _ in tqdm(range(self.args.nb_epochs)):
            optimizer.zero_grad()
            # corruption function
            idx = np.random.permutation(self.args.nb_nodes)
            features_neg = features[idx, :].to(self.args.device)

            loss_n, loss_n_v, loss_c, loss_c_v = model(features, features_pos, features_neg, adj_list, adj_pos_list, self.args.sparse)

            loss = self.coef_n * loss_n + self.coef_n_v * loss_n_v + self.coef_c * loss_c + self.coef_c_v * loss_c_v

            # early stop
            if loss < best:
                best = loss
                cnt_wait = 0
                torch.save(model.state_dict(), 'Results/epoch_{}.pkl'.format(self.args.dataset))
            else:
                cnt_wait += 1
            if cnt_wait == self.args.patience:
                print("Early stopped!")
                break

            loss.backward()
            optimizer.step()

        # save
        model.load_state_dict(torch.load('Results/epoch_{}.pkl'.format(self.args.dataset)))

        # evaluation
        print("Evaluating...")
        model.eval()
        embeds = model.embed(features, adj_list, self.args.sparse)
        macro_f1s, micro_f1s, nmi, sim = evaluate(embeds, self.idx_train, self.idx_val, self.idx_test, self.labels, self.args.dataset, )
        
        return macro_f1s, micro_f1s, nmi, sim

class modeler(nn.Module):
    def __init__(self, ft_size, hid_units, n_networks, k, t):
        super(modeler, self).__init__()
        self.k = k
        self.t = t
        self.str_mlp = Structure_mlp(hid_units)
        self.sem_mlp = Semantic_mlp(hid_units, self.k)

        self.gcn_list = nn.ModuleList([GCN(ft_size, hid_units) for _ in range(n_networks)])
        
        self.cos_r = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.cos_c = nn.CosineSimilarity(dim=0, eps=1e-6)
    
    def get_node_loss(self, h, h_pos, h_neg):
        loss = 0
        score_pos = torch.exp(self.cos_r(h, h_pos)/self.t)
        score_neg = torch.exp(self.cos_r(h, h_neg)/self.t)
        loss += -torch.sum(torch.log(score_pos/(score_pos+score_neg)))/h.shape[0]

        return loss
    
    def get_inter_node_loss(self, h_list, h_neg_list):
        loss = 0
        for i, h in enumerate(h_list):
            h_neg = h_neg_list[i]
            h_pos_list = []
            for j in range(len(h_list)):
                if i!=j:
                    h_pos_list.append(h_list[j])
            loss_v = 0
            for p in range(len(h_pos_list)):
                score_pos = torch.exp(self.cos_r(h_pos_list[p], h)/self.t)
                score_neg = torch.exp(self.cos_r(h_neg, h)/self.t)
                score = -torch.log(score_pos/(score_pos+score_neg))
                loss_v += torch.sum(score)
            loss += loss_v
        loss /= ((len(h_list)-1)*h_list[0].shape[0])
        
        return loss      
    
    def get_inter_cluster_loss(self, c_list, c_neg_list):
        loss = 0
        for i, c in enumerate(c_list):
            c_neg = c_neg_list[i]
            c_pos_list = []
            for j in range(len(c_list)):
                if i!=j:
                    c_pos_list.append(c_list[j])
            loss_v = 0
            for p in range(len(c_pos_list)):
                score_pos = torch.exp(self.cos_c(c_pos_list[p], c)/self.t)
                score_neg = torch.exp(self.cos_c(c_neg, c)/self.t)
                score = -torch.log(score_pos/(score_pos+score_neg))
                loss_v += torch.sum(score)
            loss += loss_v
        loss /= ((len(c_list)-1)*c_list[0].shape[0])
        
        return loss   
    
    def get_constraint_loss(self, c_list):
        loss = 0
        for c in c_list:
            pros = torch.sum(c, dim=0)/c.shape[0]
            loss_v = -torch.sum(torch.mul(pros, torch.log(pros)))/c.shape[1]
            loss += loss_v

        return loss
    
    def forward(self, features, features_pos, features_neg, adj_list, adj_pos_list, sparse):
        h_list = []
        h_neg_list = []
        c_list = []
        c_neg_list = []
        
        loss_n = 0
        for i, adj in enumerate(adj_list):
            # real node embedding
            h_temp = self.gcn_list[i](features, adj, sparse)
            h = torch.squeeze(self.str_mlp(h_temp))
            h_list.append(h)

            #real node semantic feature
            c = self.sem_mlp(h_temp)
            c_list.append(c)        

            # negative node embedding
            h_neg_temp = self.gcn_list[i](features_neg, adj, sparse)
            h_neg = torch.squeeze(self.str_mlp(h_neg_temp))
            h_neg_list.append(h_neg)
            c_neg = self.sem_mlp(h_neg_temp)
            c_neg_list.append(c_neg)           
            
            # positive node embedding
            h_pos_temp = self.gcn_list[i](features_pos, adj_pos_list[i], sparse)
            h_pos = torch.squeeze(self.str_mlp(h_pos_temp))
            
            loss_n += self.get_node_loss(h, h_pos, h_neg)
        
        loss_n_v = self.get_inter_node_loss(h_list, h_neg_list)
        loss_c_v = self.get_inter_cluster_loss(c_list, c_neg_list)
        loss_c = self.get_constraint_loss(c_list)
        
        return loss_n, loss_n_v, loss_c, loss_c_v

    # Detach the return variables
    def embed(self, feature, adj_list, sparse):
        h_list = []
        for i, adj in enumerate(adj_list):
            h = torch.squeeze(self.gcn_list[i](feature, adj, sparse))
            h_list.append(h)
        h_mean = torch.mean(torch.stack(h_list, 0), 0)
        
        return h_mean.detach()