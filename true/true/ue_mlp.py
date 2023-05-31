import torch.nn as nn
import torch.nn.functional as F
    
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(4096)
        self.fc1 = nn.Linear(4096, 4096)
        self.do = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.ln(x)
        x = self.do(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.do(F.relu(self.fc3(x)))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        x = F.softmax(x)
        return x
