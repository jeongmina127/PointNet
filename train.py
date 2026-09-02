import os
import torch
from data_loader import PointcloudData, default_transforms, train_transforms
import engine
import utils
from model import PointNet
from torchvision import transforms
from torch.utils.data import DataLoader

MODEL_NAME = "pointnet.pt"
NUM_EPOCHS = 100
BATCH_SIZE = 32
LOSS_WEIGHT = 0.001

LEARNING_RATE = 0.001

PATH = "/mnt/c/Users/Jeongmin Cho/Desktop/ModelNet10/ModelNet10_3_splits"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create dataloader
train_ds =  PointcloudData(root_dir=PATH,transform = train_transforms())
valid_ds = PointcloudData(root_dir=PATH,folder='val', transform= default_transforms(), valid= False)
test_ds = PointcloudData(root_dir=PATH, valid = False, folder = 'test',transform=default_transforms())

train_dataloader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_ds, batch_size = BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# Create model
model = PointNet().to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = LEARNING_RATE,
    betas=(0.9,0.999)
)

engine.train(model = model,
             train_dataloader=train_dataloader,
             valid_dataloader=valid_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device = device,
             loss_weight=LOSS_WEIGHT)


utils.save_model(model=model,
                 target_dir="models",
                 model_name =MODEL_NAME)

