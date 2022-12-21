import torch.nn as nn
import torch

class Semantic_mlp(nn.Module):
    def __init__(self, feature_dim, class_num):
        super(Semantic_mlp, self).__init__()
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.cluster_num),
            nn.Softmax(dim=-1)
        )

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        c = torch.squeeze(self.mlp(x))
        
        return c