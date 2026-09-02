import torch
from torch import nn

class PointNet(nn.Module):

    """PointNet model"""
    def __init__(self):
        super().__init__()
        self.tnet1 = TNet1()
        self.tnet2 = TNet2()
        self.f0 = nn.Conv1d(3,64,1)
        self.f1 = nn.Conv1d(64,64,1)

        self.f2 = nn.Conv1d(64, 128, 1)
        self.f3 = nn.Conv1d(128,1024,1)

        self.f4 = nn.Linear(1024,512)
        self.f5 = nn.Linear(512,256)

        self.dropout = nn.Dropout(p=0.3)

        self.f6 = nn.Linear(256,10)

    def forward(self,x):
        A = self.tnet1(x)
        print("noting:", x.shape)
        x = torch.bmm(A, x)
        print("after T1:", x.shape)
        x = self.f0(x)
        print("f0:", x.shape)
        x = self.f1(x)
        print("f1:", x.shape)    
        A = self.tnet2(x)
        x = torch.bmm(A, x)
        print("after T2:", x.shape)    
        x = self.f2(x)
        print("f2:", x.shape)    
        x = self.f3(x)
        print("f3:", x.shape)    
        x = torch.max(x, dim = 2).values
        print("torch.max:", x.shape)    
        x = self.f4(x)
        print("f4:", x.shape)    
        x = self.f5(x)
        print("f5:", x.shape)    
        x = self.dropout(x)
        print("after dropout:", x.shape)    
        x = self.f6(x)
        print("f6:", x.shape)    

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

            x = x.reshape(-1, 64, 64)

            return x