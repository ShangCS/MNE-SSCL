import torch.nn as nn
import torch

class Structure_mlp(nn.Module):
    def __init__(self, feature_dim):
        super(Structure_mlp, self).__init__()
        self.feature_dim = feature_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU()#ACM and Amazon: ReLU; IMDB: Tanh
        )
        
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    
    def forward(self, x):
        out = torch.squeeze(self.mlp(x))

        return out
