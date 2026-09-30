from data_loader import default_transforms, PointcloudData
from engine import feature_transform_regularizer
from model import PointNet

from pathlib import Path
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


RUN_ID = ""
DATA_PATH = "/mnt/c/Users/Jeongmin Cho/Desktop/ModelNet10/ModelNet10_3_splits"
MODEL_DIR = Path("results") / RUN_ID
LOSS_WEIGHT = 0.001
BATCH_SIZE = 32
LOSS_WEIGHT = 0.001

device = torch.device("cpu")
loss_fn = torch.nn.CrossEntropyLoss()

test_ds = PointcloudData(root_dir=DATA_PATH, valid = False, folder = 'test',transform=default_transforms())
test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


model = PointNet().to(device)
model.load_state_dict(torch.load(MODEL_DIR / 'best.pt', map_location=device, weights_only=True))


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              device: torch.device,
              loss_fn: torch.nn.Module,
              loss_weight: float) -> Tuple[list, list, float, float]:

    all_y_true = []
    all_y_pred = []
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    model.eval()

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc='Testing', ncols=80, colour="red", leave=False):

            X = batch["pointcloud"]
            y = batch["category"]

            #GPT correction
            X = X.transpose(1, 2)

            X, y = X.float().to(device), y.long().view(-1).to(device)

            test_pred_logits, A = model(X)

            classfication_loss= loss_fn(test_pred_logits, y)
            regularization_loss = feature_transform_regularizer(A)
            loss = classfication_loss + loss_weight * regularization_loss

            batch_size = y.size(0)

            total_loss += loss.item() * batch_size

            test_pred_labels = test_pred_logits.argmax(dim=1)
            all_y_pred.extend(test_pred_labels.cpu().tolist())
            all_y_true.extend(y.cpu().tolist())

            total_correct += (test_pred_labels == y).sum().item()

            total_samples += batch_size

        test_loss = total_loss / total_samples
        test_acc = total_correct / total_samples

    return all_y_true, all_y_pred, test_loss, test_acc

all_y_true, all_y_pred, test_loss, test_acc = test_step(model = model,
                                                        dataloader=test_dataloader,
                                                        device=device,
                                                        loss_fn=loss_fn,
                                                        loss_weight=LOSS_WEIGHT)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")


class_names = [
    name
    for name, idx in sorted(
        test_ds.classes.items(),
        key=lambda x: x[1]
    )
]
class_ids = list(range(len(class_names)))
cm = confusion_matrix(all_y_true, all_y_pred, labels = class_ids)


# Confusion matrix 그림 생성
fig, ax = plt.subplots(figsize=(10, 10))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    xticks_rotation=45
)

ax.set_title("Test Confusion Matrix")

fig.tight_layout()

# 현재 run 폴더에 저장
save_path = MODEL_DIR / "confusion_matrix.png"

fig.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(f"Confusion matrix saved to: {save_path}")

