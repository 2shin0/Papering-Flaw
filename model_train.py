import torch
import torch.nn as nn
import torchvision.models as models
from setting import CFG
import tqdm
import numpy as np
from sklearn.metrics import f1_score
import torch.optim as optim


def train_h(model, optimizer, train_loader, val_loader, scheduler, device, patience=3):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    best_score = 0
    best_model = None
    
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for imgs, labels in tqdm.tqdm(train_loader):
            imgs = imgs.float().to(device)
            labels = labels.long().to(device)
            
            optimizer.zero_grad()
            
            output = model(imgs)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val Weighted F1 Score : [{_val_score:.5f}]')
       
        if scheduler is not None:
            scheduler.step(_val_score)
            
        # Early Stopping
        if _val_score <= best_score:
            counter += 1
            if counter >= patience:
                print(f'Validation performance did not improve for {patience} epochs. Early stopping...')
                break
        else:
            best_score = _val_score
            best_model = model
            counter = 0  # 카운터 초기화
    
    return best_model




def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, true_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm.tqdm(val_loader):
            imgs = imgs.float().to(device)
            labels = labels.long().to(device)
            
            pred = model(imgs)
            
            loss = criterion(pred, labels)
            
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += labels.detach().cpu().numpy().tolist()
            
            val_loss.append(loss.item())
        
        _val_loss = np.mean(val_loss)
        _val_score = f1_score(true_labels, preds, average='weighted')
    
    return _val_loss, _val_score
