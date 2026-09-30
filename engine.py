import torch
import pandas as pd
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

def feature_transform_regularizer(A):
    I = torch.eye(A.size(1), device = A.device, dtype=A.dtype).unsqueeze(0)
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

    total_loss = 0.0
    total_correct = 0
    total_samples = 0


    for batch in tqdm(dataloader, desc='Training', ncols=80, colour="blue", leave=False):

        X = batch["pointcloud"]
        y = batch["category"]

        #GPT correction        
        X = X.transpose(1, 2)

        X, y = X.float().to(device), y.long().view(-1).to(device)   ## CHECK

        optimizer.zero_grad()
        y_pred_logits, A = model(X)

        classfication_loss= loss_fn(y_pred_logits, y)
        regularization_loss = feature_transform_regularizer(A)
        loss = classfication_loss + loss_weight * regularization_loss


        loss.backward()
        optimizer.step()

        batch_size = y.size(0)

        total_loss += loss.item() * batch_size

        y_pred_class = y_pred_logits.argmax(dim=1)

        total_correct += (y_pred_class == y).sum().item()
        total_samples += batch_size

        # precision, recall = calc_precision_recall(y = y, y_pred_logits= y_pred_logits)
        # precision_list.append(precision)
        # recall_list.append(recall)

    train_loss = total_loss / total_samples
    train_acc = total_correct / total_samples
    return train_loss, train_acc


def valid_step(model : torch.nn.Module,
              dataloader : torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device : torch.device,
              loss_weight : float) -> Tuple[float, float]:

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc='Validating', ncols=80, colour="yellow", leave=False):

            X = batch["pointcloud"]
            y = batch["category"]

            #GPT correction
            X = X.transpose(1, 2)

            X, y = X.float().to(device), y.long().view(-1).to(device)

            val_pred_logits, A = model(X)

            classfication_loss= loss_fn(val_pred_logits, y)
            regularization_loss = feature_transform_regularizer(A)
            loss = classfication_loss + loss_weight * regularization_loss

            batch_size = y.size(0)

            total_loss += loss.item() * batch_size

            val_pred_labels = val_pred_logits.argmax(dim=1)

            total_correct += (val_pred_labels == y).sum().item()

            total_samples += batch_size

        val_loss = total_loss / total_samples
        val_acc = total_correct / total_samples
    return val_loss, val_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          valid_dataloader : torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn : torch.nn.Module,
          epochs: int,
          device : torch.device,
          loss_weight : float,
          early_stopping,
          run_dir) -> Dict[str, List]:
    
    results = {"train_loss":[],
               "train_acc":[],
               "val_loss":[],
               "val_acc":[]}

    schedular = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.5
    )
    writer = SummaryWriter(log_dir=str(Path(run_dir) / "tensorboard"))
    # ChatGPT's code
    sample_batch = next(iter(train_dataloader))
    sample_X = sample_batch["pointcloud"]
    # [B, N, C] -> [B, C, N]
    sample_X = sample_X.transpose(1, 2)
    sample_X = sample_X.float().to(device)
    model.eval()
    ## END
    writer.add_graph(model=model, input_to_model=sample_X)
    for epoch in range(epochs):
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
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        writer.add_scalars(main_tag = "Loss",
                           tag_scalar_dict={"train_loss":train_loss,
                                            "val_loss":val_loss},
                           global_step=epoch+ 1)

        writer.add_scalars(main_tag = "Accuracy",
                           tag_scalar_dict={"train_acc":train_acc,
                                            "val_acc":val_acc},
                           global_step=epoch+ 1)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    writer.close()
    pd.DataFrame(results).to_csv(Path(run_dir)/"training_results.csv", index=False)

    return results