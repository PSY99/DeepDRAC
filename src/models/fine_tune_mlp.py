import torch
from torch import nn
class Fine_tune_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Fine_tune_MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size*2)
        self.fc2 = nn.Linear(hidden_size*2, hidden_size*2)
        self.fc3 = nn.Linear(hidden_size*2, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x