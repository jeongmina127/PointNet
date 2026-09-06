import torch
from torch import nn

class PointNet(nn.Module):

    """PointNet model"""
    def __init__(self):
        super().__init__()
        self.tnet1 = TNet1()
        self.tnet2 = TNet2()

        self.f0 = nn.Conv1d(3,64,1)
        self.bn_f0 = nn.BatchNorm1d(64)

        self.f1 = nn.Conv1d(64,64,1)
        self.bn_f1 = nn.BatchNorm1d(64)

        self.f2 = nn.Conv1d(64, 128, 1)
        self.bn_f2 = nn.BatchNorm1d(128)

        self.f3 = nn.Conv1d(128,1024,1)
        self.bn_f3 = nn.BatchNorm1d(1024)

        self.f4 = nn.Linear(1024,512)
        self.bn_f4 = nn.BatchNorm1d(512)

        self.f5 = nn.Linear(512,256)
        self.bn_f5 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        self.f6 = nn.Linear(256,10)

    def forward(self,x):
        A = self.tnet1(x)
        x = torch.bmm(A, x)

        x = self.f0(x)
        x = self.bn_f0(x)
        x = self.relu(x)

        x = self.f1(x)
        x = self.bn_f1(x)
        x = self.relu(x)

        A = self.tnet2(x)
        x = torch.bmm(A, x)

        x = self.f2(x)
        x = self.bn_f2(x)
        x = self.relu(x)

        x = self.f3(x)
        x = self.bn_f3(x)
        x = self.relu(x)

        x = torch.max(x, dim = 2).values

        x = self.f4(x)
        x = self.bn_f4(x)
        x = self.relu(x)

        x = self.f5(x)
        x = self.bn_f5(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.f6(x)

        return x,A



class TNet1(nn.Module): 

        def __init__(self):
            super().__init__()
            self.f1 = nn.Conv1d(3,64,1)
            self.bn_f1 = nn.BatchNorm1d(num_features=64)
            
            self.f2 = nn.Conv1d(64,128,1)
            self.bn_f2 = nn.BatchNorm1d(num_features=128)
            self.relu = nn.ReLU()
            
            self.f3 = nn.Conv1d(128,1024,1)
            self.bn_f3 = nn.BatchNorm1d(num_features=1024)
            
            self.f4 = nn.Conv1d(1024,512,1)
            self.bn_f4 = nn.BatchNorm1d(num_features=512)
            
            self.f5 = nn.Linear(512,256)
            self.f6 = nn.Linear(256, 9)

            # GPT correction
            nn.init.zeros_(self.f6.weight)
            nn.init.zeros_(self.f6.bias)


        def forward(self,x):
            x = self.f1(x)
            x = self.bn_f1(x)
            x = self.relu(x)

            x = self.f2(x)
            x = self.bn_f2(x)
            x = self.relu(x)

            x = self.f3(x)
            x = self.bn_f3(x)
            x = self.relu(x)

            x = self.f4(x)
            x = self.bn_f4(x)
            x = self.relu(x)

            x = torch.max(x, dim = 2).values

            x = self.f5(x)
            x = self.f6(x)

            # GPT correction
            identity = torch.eye(3, device=x.device, dtype=x.dtype).reshape(1, 9)
            x = x + identity

            x = x.reshape(-1, 3, 3)

            return x

class TNet2(nn.Module): 

        def __init__(self):
            super().__init__()
            self.f1 = nn.Conv1d(64,64,1)
            self.bn_f1 = nn.BatchNorm1d(num_features=64)
            
            self.f2 = nn.Conv1d(64,128,1)
            self.bn_f2 = nn.BatchNorm1d(num_features=128)
            self.relu = nn.ReLU()
            
            self.f3 = nn.Conv1d(128,1024,1)
            self.bn_f3 = nn.BatchNorm1d(num_features=1024)
            
            self.f4 = nn.Conv1d(1024,512,1)
            self.bn_f4 = nn.BatchNorm1d(num_features=512)
            
            self.f5 = nn.Linear(512,256)
            self.f6 = nn.Linear(256, 4096)

            # GPT correction
            nn.init.zeros_(self.f6.weight)
            nn.init.zeros_(self.f6.bias)


        def forward(self,x):
            x = self.f1(x)
            x = self.bn_f1(x)
            x = self.relu(x)

            x = self.f2(x)
            x = self.bn_f2(x)
            x = self.relu(x)

            x = self.f3(x)
            x = self.bn_f3(x)
            x = self.relu(x)

            x = self.f4(x)
            x = self.bn_f4(x)
            x = self.relu(x)

            x = torch.max(x, dim = 2).values

            x = self.f5(x)
            x = self.f6(x)

            # GPT correction
            identity = torch.eye(64, device=x.device, dtype=x.dtype).reshape(1, 4096)
            x = x + identity

            x = x.reshape(-1, 64, 64)

            return x