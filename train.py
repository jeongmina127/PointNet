import os
from datetime import datetime
from pathlib import Path
import torch
from data_loader import PointcloudData, default_transforms, train_transforms
import engine
import utils
from model import PointNet
from early_stopping_pytorch import EarlyStopping
from torch.utils.data import DataLoader

MODEL_NAME = "pointnet.pt"
NUM_EPOCHS = 1000
BATCH_SIZE = 32
LOSS_WEIGHT = 0.001
NUM_WORKERS = 0 # Due to CPU
LEARNING_RATE = 0.001
PATIENCE = 25

PATH = "/mnt/c/Users/noork/Documents/Classification3D/ModelNet10_3_splits"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
RUN_DIR = Path("models") / f"run_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=False)
print(f"Saving this training run to: {RUN_DIR}")
CHECKPOINT_PATH = RUN_DIR / "best.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
early_stopping = EarlyStopping(patience = PATIENCE, verbose = True, path = CHECKPOINT_PATH)

# Create dataloader
train_ds =  PointcloudData(root_dir=PATH,transform = train_transforms())
valid_ds = PointcloudData(root_dir=PATH,folder='val', transform= default_transforms(), valid= False)
test_ds = PointcloudData(root_dir=PATH, valid = False, folder = 'test',transform=default_transforms())

train_dataloader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle=True,num_workers=NUM_WORKERS)
valid_dataloader = DataLoader(valid_ds, batch_size = BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS)
test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS)

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
             loss_weight=LOSS_WEIGHT,
             early_stopping = early_stopping)

model.load_state_dict(torch.load(CHECKPOINT_PATH,
                                 map_location=device,
                                 weights_only=True))

utils.save_model(model=model,
                 target_dir=RUN_DIR,
                 model_name =MODEL_NAME)

