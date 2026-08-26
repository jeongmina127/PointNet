import torch
from torch import nn

class PointNet(nn.Module):

    """PointNet model"""
    def __init__(self):
        super().__init__()
        self.tnet1 = TNet1
        self.tnet2 = TNet2
        self.f1 = nn.Linear(64,64)

        self.f2 = nn.Linear(64, 128)
        self.f3 = nn.Linear(128,1024)

        self.f4 = nn.Linear(1024,512)
        self.f5 = nn.Linear(512,256)
        self.f6 = nn.Linear(256,10)

    def forward(self,x):
        x = self.tnet1(x)
        x = self.f1(x)
        x = self.tnet2(x)
        x = self.f2(x)
        x = self.f3(x)
        x = torch.max(x, dim = 1)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)



class TNet1(nn.Module): 

        def __init__(self):
            super().__init__()
            self.f1 = nn.Linear(3,64)
            self.bn_fc1 = nn.BatchNorm1d(num_features=64)
            
            self.f2 = nn.Linear(64,128)
            self.bn_fc2 = nn.BatchNorm1d(num_features=128)
            self.relu = nn.ReLU()
            
            self.f3 = nn.Linear(128,1024)
            self.bn_fc3 = nn.BatchNorm1d(num_features=1024)
            
            self.f4 = nn.Linear(1024,512)
            self.bn_fc4 = nn.BatchNorm1d(num_features=512)
            
            self.f5 = nn.Linear(512,256)
            self.f6 = nn.Linear(256, 9)


        def forward(self,x):
            x = self.f1(x)
            x = self.bn_f1(x)
            x = nn.ReLU()

            x = self.f2(x)
            x = self.bn_f2(x)
            x = nn.ReLU()

            x = self.f3(x)
            x = self.bn_f3(x)
            x = nn.ReLU()

            x = self.f4(x)
            x = self.bn_f4(x)
            x = nn.ReLU()

            x = torch.max(x, dim = 1)

            x = self.f5(x)
            x = self.f6(x)


class TNet2(nn.Module): 

        def __init__(self):
            super().__init__()
            self.f1 = nn.Linear(64,64)
            self.bn_fc1 = nn.BatchNorm1d(num_features=64)
            
            self.f2 = nn.Linear(64,128)
            self.bn_fc2 = nn.BatchNorm1d(num_features=128)
            self.relu = nn.ReLU()
            
            self.f3 = nn.Linear(128,1024)
            self.bn_fc3 = nn.BatchNorm1d(num_features=1024)
            
            self.f4 = nn.Linear(1024,512)
            self.bn_fc4 = nn.BatchNorm1d(num_features=512)
            
            self.f5 = nn.Linear(512,256)
            self.f6 = nn.Linear(256, 4096)


        def forward(self,x):
            x = self.f1(x)
            x = self.bn_f1(x)
            x = nn.ReLU()

            x = self.f2(x)
            x = self.bn_f2(x)
            x = nn.ReLU()

            x = self.f3(x)
            x = self.bn_f3(x)
            x = nn.ReLU()

            x = self.f4(x)
            x = self.bn_f4(x)
            x = nn.ReLU()

            x = torch.max(x, dim = 1)

            x = self.f5(x)
            x = self.f6(x)
