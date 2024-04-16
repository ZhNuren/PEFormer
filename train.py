from radfusion import RadFusionCT
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import yaml
import wandb
import json
import pytorch_warmup as warmup


config = yaml.load(open("config.yaml", 'r'), Loader=yaml.FullLoader)
print(config)
run = wandb.init(entity='biomed', project='hc_project', config=config)


def plot_results(results, save_dir, name = None):

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(results['train_loss'], label='Train loss')
    plt.plot(results['val_loss'], label='Validation loss')

    
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(results['train_acc'], label='Train accuracy')
    plt.plot(results['val_acc'], label='Validation accuracy')
    plt.plot(results['train_f1'], label='Train F1')
    plt.plot(results['train_recall'], label='Train Recall')
    
    plt.plot(results['val_f1'], label='Val F1')
    plt.plot(results['val_recall'], label='Val Recall')
    plt.legend()
    if name:
        plt.savefig(save_dir + name)
    else:
        plt.savefig(save_dir + '/LossAccuracy.png')

class AbsolutePositionalEncoder(nn.Module):
    def __init__(self, emb_dim, max_position=111):
        super(AbsolutePositionalEncoder, self).__init__()
        self.position = torch.arange(max_position).unsqueeze(1)

        self.positional_encoding = torch.zeros(1, max_position, emb_dim)

        _2i = torch.arange(0, emb_dim, step=2).float()

        # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        self.positional_encoding[0, :, 0::2] = torch.sin(self.position / (10000 ** (_2i / emb_dim)))

        # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        self.positional_encoding[0, :, 1::2] = torch.cos(self.position / (10000 ** (_2i / emb_dim)))

    def forward(self, x):
        # batch_size, input_len, embedding_dim
        batch_size, seq_len, _ = x.size()

        return self.positional_encoding[:batch_size, :seq_len, :]



class PEFormer(nn.Module):
    def __init__(self, 
                 max_num_emb=110, 
                 emb_dim = 2048, 
                 num_heads = 2,
                 num_enc_layer = 1,
                 num_classes = 1,
                 dim_feedforward = 512):
        super(PEFormer, self).__init__()

        self.positional_encoder = AbsolutePositionalEncoder(emb_dim, max_position=max_num_emb+1)

        self.cl_token = nn.Parameter(torch.randn(1, emb_dim))

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_enc_layer)

        self.fc = nn.Linear(emb_dim, num_classes)
    
    def forward(self, x, mask):
        batch_size, seq_len, emb_dim = x.size()

        x = torch.cat([self.cl_token.expand(batch_size, -1, -1), x], dim=1)
        pos_enc = self.positional_encoder(x).to(x.device)
        x = x + pos_enc

        #add False to the beginning of mask to account for CL token
        mask = torch.cat([torch.tensor([[False]]).expand(batch_size, 1).to(mask.device), mask], dim=1)

        x = self.transformer(x, src_key_padding_mask = mask)

        x = self.fc(x[:, 0, :])

        return x

def train_step(
        model: torch.nn.Module,
        train_loader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        lr_scheduler: torch.optim.lr_scheduler,
        warmup_scheduler: warmup.LinearWarmup,
        warmup_period: int,
):
    """
    Train model for one epoch.

    Args:
        model: PyTorch model to train.
        train_loader: PyTorch dataloader for training data.
        loss_fn: PyTorch loss function.
        optimizer: PyTorch optimizer.
        device: PyTorch device to use for training.

    Returns:
        Average loss, accuracy, macro F1 score, and macro recall for the epoch.
    """

    model.train()
    train_loss = 0.0

    targets = []
    predictions = []

    for i, (_, emb, mask, target) in enumerate(tqdm(train_loader)):
        target = target.float()
        emb, mask, target = emb.to(device), mask.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(emb, mask)     
        output = output.squeeze(1) 
        
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        with warmup_scheduler.dampening():
            if warmup_scheduler.last_step + 1 >= warmup_period:
                lr_scheduler.step()

        train_loss += loss.item()

        prediction = torch.sigmoid(output) > 0.5

        predictions.extend(prediction.cpu().numpy())
        targets.extend(target.cpu().numpy())



    train_loss /= len(train_loader)
    train_acc = accuracy_score(targets, predictions)
    train_macro_f1 = f1_score(targets, predictions, average='macro')
    train_macro_recall = recall_score(targets, predictions, average='macro')
    train_precision = precision_score(targets, predictions, average='macro')
    train_auc = roc_auc_score(targets, predictions)

    return train_loss, train_acc, train_macro_f1, train_macro_recall, train_precision, train_auc

def val_step(
        model: torch.nn.Module,
        val_loader,
        loss_fn: torch.nn.Module,
        device: torch.device,
):
    """
    Evaluate model on val data.

    Args:
        model: PyTorch model to evaluate.
        val_loader: PyTorch dataloader for val data.
        loss_fn: PyTorch loss function.
        device: PyTorch device to use for evaluation.

    Returns:
        Average loss, accuracy, macro F1 score, and macro recall for the validation set.
    """

    model.eval()
    val_loss = 0.0
    val_targets = []
    val_predictions = []


    with torch.no_grad():
        for i, (_, emb, mask, target) in enumerate(tqdm(val_loader)):
            target = target.float()
            emb, mask, target = emb.to(device), mask.to(device), target.to(device)
            output = model(emb, mask)
            output = output.squeeze(1)
            loss = loss_fn(output, target)
            val_loss += loss.item()

            prediction = torch.sigmoid(output) > 0.5


            val_predictions.extend(prediction.cpu().numpy())
            val_targets.extend(target.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_targets, val_predictions)
        val_macro_f1 = f1_score(val_targets, val_predictions, average='macro')
        val_macro_recall = recall_score(val_targets, val_predictions, average='macro')
        val_precision = precision_score(val_targets, val_predictions, average='macro')
        val_auc = roc_auc_score(val_targets, val_predictions)



    return val_loss, val_acc, val_macro_f1, val_macro_recall, val_precision, val_auc


def test_step(
        model: torch.nn.Module,
        test_loader,
        loss_fn: torch.nn.Module,
        device: torch.device,
):
    """
    Evaluate model on test data.

    Args:
        model: PyTorch model to evaluate.
        test_loader: PyTorch dataloader for test data.
        loss_fn: PyTorch loss function.
        device: PyTorch device to use for evaluation.

    Returns:
        Average loss, accuracy, macro F1 score, and macro recall for the test set.
    """

    model.eval()
    test_loss = 0.0
    test_targets = []
    test_predictions = []

    with torch.no_grad():
        for i, (_, emb, mask, target) in enumerate(tqdm(test_loader)):
            target = target.float()
            emb, mask, target = emb.to(device), mask.to(device), target.to(device)
            output = model(emb, mask)
            output = output.squeeze(1)
            loss = loss_fn(output, target)
            test_loss += loss.item()

            prediction = torch.sigmoid(output) > 0.5

            test_predictions.extend(prediction.cpu().numpy())
            test_targets.extend(target.cpu().numpy())

        test_loss /= len(test_loader)
        test_acc = accuracy_score(test_targets, test_predictions)
        test_macro_f1 = f1_score(test_targets, test_predictions, average='macro')
        test_macro_recall = recall_score(test_targets, test_predictions, average='macro')
        test_precision = precision_score(test_targets, test_predictions, average='macro')
        test_auc = roc_auc_score(test_targets, test_predictions)

    return test_loss, test_acc, test_macro_f1, test_macro_recall, test_precision, test_auc

def trainer(
        model: torch.nn.Module,
        train_loader,
        val_loader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler,
        warmup_scheduler: warmup.LinearWarmup,
        warmup_period: int,
        lr_scheduler_name: str,
        device: torch.device,
        epochs: int,
        save_dir: str,
        early_stopper=None,
        start_epoch = 1,
):
    """
    Train and evaluate model.

    Args:
        model: PyTorch model to train.
        train_loader: PyTorch dataloader for training data.
        val_loader: PyTorch dataloader for val data.
        loss_fn: PyTorch loss function.
        optimizer: PyTorch optimizer.
        lr_scheduler: PyTorch learning rate scheduler.
        device: PyTorch device to use for training.
        epochs: Number of epochs to train the model for.

    Returns:
        Average loss and accuracy for the val set.
    """

    results = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1":[],
        "val_f1":[],
        "train_recall":[],
        "val_recall":[],
        "train_precision":[],
        "val_precision":[],
        "train_auc":[],
        "val_auc":[]

    }
    best_val_loss = 1e10
    test_every = 2
    for epoch in range(start_epoch, epochs + 1):

        print(f"Epoch {epoch}:")
        train_loss, train_acc, train_macro_f1, train_macro_recall, train_precision, train_auc = train_step(model, train_loader, loss_fn, optimizer, device, lr_scheduler, warmup_scheduler, warmup_period)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_macro_f1:.4f}, Train recall: {train_macro_recall:.4f}, Train precision: {train_precision:.4f}, Train AUC: {train_auc:.4f}")

        

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_f1"].append(train_macro_f1)
        results["train_recall"].append(train_macro_recall)
        results["train_precision"].append(train_precision)
        results["train_auc"].append(train_auc)

        val_loss, val_acc, val_f1, val_recall, val_precision, val_auc = val_step(model, val_loader, loss_fn, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val recall: {val_recall:.4f}, Val precision: {val_precision:.4f}, Val AUC: {val_auc:.4f}")
        print()

        if epoch % test_every == 0:
            test_loss, test_acc, test_f1, test_recall, test_precision, test_auc = test_step(model, test_loader, loss_fn, device)
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test recall: {test_recall:.4f}, Test precision: {test_precision:.4f}, Test AUC: {test_auc:.4f}")
            print()
        # if lr_scheduler_name == "ReduceLROnPlateau":
        #     lr_scheduler.step(val_loss)
        # elif lr_scheduler_name != "None":
        #     lr_scheduler.step()
        
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["val_f1"].append(val_f1)
        results["val_recall"].append(val_recall)
        results["val_precision"].append(val_precision)
        results["val_auc"].append(val_auc)
        
        
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, 
                   "train_acc": train_acc, "val_acc": val_acc, 
                   "train_f1": train_macro_f1, "val_f1": val_f1,
                   "train_recall": train_macro_recall, "val_recall": val_recall,
                   "train_precision": train_precision, "val_precision": val_precision,
                   "train_auc": train_auc, "val_auc": val_auc})
        

        checkpoint = { 
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_sched': lr_scheduler}
            
                    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(save_dir, "best_checkpoint.pth"))

        torch.save(checkpoint, os.path.join(save_dir, "last_checkpoint.pth"))

        if early_stopper is not None:
            if early_stopper.early_stop(val_loss):
                print("Early stopping")
                break

    return results

def START_seed(seed_value=9):
    seed = seed_value
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 


BATCH_SIZE = int(config["BATCH_SIZE"])
LEARNING_RATE = float(config["LEARNING_RATE"])
NUM_EPOCHS = int(config["NUM_EPOCHS"])
LEARNING_SCHEDULER = config["LEARNING_SCHEDULER"]
LOSS = config["LOSS"]
SAVE_DIR = config["SAVE_DIR"]
MIN_LR = float(config["MIN_LR"])
WARMUP_EPOCH = int(config["WARMUP_EPOCH"])
DEVICE_NUM = int(config["DEVICE_NUM"])
DEVICE = torch.device(f"cuda:{DEVICE_NUM}" if torch.cuda.is_available() else 'cpu')

print(f"Using {DEVICE} device")


all_ct_embs = config["CT_PK"]
all_emr_embs = config["EHR_PK"]
labels_csv = config["CSV"]


dataset_train = RadFusionCT(pkl_ct_file=all_ct_embs, pkl_emr_file=all_emr_embs, csv_file=labels_csv, split='train')
dataset_val = RadFusionCT(pkl_ct_file=all_ct_embs, pkl_emr_file=all_emr_embs, csv_file=labels_csv, split='val')
dataset_test = RadFusionCT(pkl_ct_file=all_ct_embs, pkl_emr_file=all_emr_embs, csv_file=labels_csv, split='test')


train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

START_seed()
run_id = time.strftime("%Y-%m-%d_%H-%M-%S")

save_dir = os.path.join(SAVE_DIR, run_id)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model = PEFormer()
model.to(DEVICE)

torch.compile(model)

if LOSS == "MSE":
    loss = torch.nn.MSELoss()
elif LOSS == "L1Loss":
    loss = torch.nn.L1Loss()
elif LOSS == "SmoothL1Loss":
    loss = torch.nn.SmoothL1Loss()
elif LOSS == "CrossEntropyLoss":
    loss = torch.nn.CrossEntropyLoss()
elif LOSS == "BCEWithLogitsLoss":
    loss = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

warmup_period = WARMUP_EPOCH * len(train_loader)

num_steps = len(train_loader) * NUM_EPOCHS - warmup_period

if LEARNING_SCHEDULER == "CosineAnnealingLR":
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, verbose=False, eta_min=MIN_LR)
elif LEARNING_SCHEDULER == "ReduceLROnPlateau":
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
elif LEARNING_SCHEDULER == "StepLR":
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)
elif LEARNING_SCHEDULER == "MultiStepLR":
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 20], gamma=0.1)
else:
    lr_scheduler = None

warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)

results = trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        warmup_scheduler=warmup_scheduler,
        warmup_period=warmup_period,
        lr_scheduler_name=LEARNING_SCHEDULER,
        device=DEVICE,
        epochs=NUM_EPOCHS,
        save_dir=save_dir,
        early_stopper=None,
    )

checkpoint = torch.load(os.path.join(save_dir, "best_checkpoint.pth"))
model.load_state_dict(checkpoint['model'])
model.to(DEVICE)
torch.compile(model)

test_loss, test_acc, test_f1, test_recall = test_step(model, test_loader, loss_fn=loss, device = DEVICE)
print(test_loss, test_acc, test_f1, test_recall)
config["test_acc"] = test_acc
config["test_loss"] = test_loss
config["test_f1"] = test_f1
config["test_recall"] = test_recall

wandb.log({"test_loss": test_loss, "test_acc": test_acc, "test_f1": test_f1, "test_recall": test_recall})

train_summary = {
    "config": config,
    "results": results,
}

with open(os.path.join(save_dir, "train_summary.json"), "w") as f:
    json.dump(train_summary, f, indent=4)


plot_results(results, save_dir)  
