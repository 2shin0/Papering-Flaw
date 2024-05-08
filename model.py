import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

import numpy as np

from sklearn.metrics import f1_score
from tqdm.auto import tqdm

from preprocessing import CFG, le, train_loader, val_loader

cuda_available = torch.cuda.is_available()

if cuda_available:
    num_cuda_devices = torch.cuda.device_count()
    cuda_device_name = torch.cuda.get_device_name(0)  
    
    print("CUDA 사용 가능:", cuda_available)
    print("CUDA 장치 수:", num_cuda_devices)
    print("사용 중인 CUDA 장치:", cuda_device_name)
else:
    print("CUDA를 사용할 수 없습니다.")

device = torch.device('cpu') if torch.cuda.is_available() else torch.device('cpu')

class BaseModel(nn.Module):
    def __init__(self, num_classes=len(le.classes_)):
        super(BaseModel, self).__init__()
        self.backbone = models.efficientnet_b4(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    
def train(model, optimizer, train_loader, val_loader, scheduler, device, patience=3):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    best_score = 0
    best_model = None
    
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)):
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
            
        if _val_score <= best_score:
            counter += 1
            if counter >= patience:
                print(f'Validation performance did not improve for {patience} epochs. Early stopping...')
                break
        else:
            best_score = _val_score
            best_model = model
            counter = 0  
    
    return best_model

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, true_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
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

def inference(model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)
            
            pred = model(imgs)
            
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
    
    preds = le.inverse_transform(preds)
    return preds