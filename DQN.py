from torch import nn
import torch.nn.functional as F

class deep_q_network(nn.Module):
    def __init__(self, input_size, output_size):
        super(deep_q_network, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.ln1 = nn.LayerNorm(128)
        self.layer2 = nn.Linear(128, 128)
        self.ln2 = nn.LayerNorm(128)
        self.layer3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.ln1(x)
        x = F.relu(x)
        
        x = self.layer2(x)
        x = self.ln2(x)
        x = F.relu(x)
        
        x = self.layer3(x)
        return x

    def predict(self, state):
        return self.forward(state)
    
