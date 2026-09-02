import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple


def feature_transform_regularizer(A):
    I = torch.eye(A.size(1), device = A.device).unsqueeze(0)
    AAT = torch.bmm(A, A.transpose(1,2))
    loss = ((I - AAT)**2).sum(dim=(1,2)).mean()
    return loss



def train_step(model : torch.nn.Module,
               dataloader : torch.utils.data.DataLoader,
               loss_fn : torch.nn.Module,
               optimizer : torch.optim.Optimizer,
               device : torch.device,
               loss_weight : float) -> Tuple[float, float]:
    model.train()

    train_loss, train_acc = 0,0

    for batch_idx, batch in enumerate(dataloader):

        X = batch["pointcloud"]
        y = batch["category"]

        #GPT correction        
        #X = X.transpose(1, 2)

        X, y = X.float().to(device), y.long().to(device)

        y_pred, A = model(X)

        classfication_loss= loss_fn(y_pred, y)
        regularization_loss = feature_transform_regularizer(A)
        loss = classfication_loss + loss_weight * regularization_loss

        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim = 1), dim = 1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def valid_step(model : torch.nn.Module,
              dataloader : torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device : torch.device,
              loss_weight : float) -> Tuple[float, float]:

    model.eval()

    val_loss, val_acc = 0,0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):

            X = batch["pointcloud"]
            y = batch["category"]

            #GPT correction
            #X = X.transpose(1, 2)

            X, y = X.float().to(device), y.long().to(device)

            val_pred_logits, A = model(X)

            classfication_loss= loss_fn(val_pred_logits, y)
            regularization_loss = feature_transform_regularizer(A)
            loss = classfication_loss + loss_weight * regularization_loss

            val_loss += loss.item()

            val_pred_labels = val_pred_logits.argmax(dim = 1)
            val_acc += ((val_pred_labels == y).sum().item()/len(val_pred_labels)) 

        val_loss = val_loss / len(dataloader)
        val_acc = val_acc / len(dataloader)
        return val_loss, val_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          valid_dataloader : torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn : torch.nn.Module,
          epochs: int,
          device : torch.device,
          loss_weight : float) -> Dict[str, List]:
    
    results = {"train_loss":[],
               "train_acc":[],
               "val_loss":[],
               "val_acc":[]}
    
    schedular = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.5
    )

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model = model,
                                           dataloader = train_dataloader,
                                           loss_fn = loss_fn,
                                           optimizer = optimizer,
                                           device = device,
                                           loss_weight=loss_weight)
        val_loss, val_acc = valid_step(model = model,
                                       dataloader = valid_dataloader,
                                       loss_fn = loss_fn,
                                       device = device,
                                       loss_weight=loss_weight)
        schedular.step()
        
        print(
            f"Epoch: {epoch+1} |"
            f"train_loss: {train_loss:.4f} |"
            f"train_acc: {train_acc:.4f} |"
            f"val_loss: {val_loss:.4f} |"
            f"val_acc: {val_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

    return results