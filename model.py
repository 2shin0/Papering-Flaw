import torch
import torch.nn as nn
import torchvision.models as models
from setting import CFG
import torch.optim as optim
from model_train import train

class BaseModel(nn.Module):
    def __init__(self, num_classes=19):
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

def model_make(train_loader, val_loader, device):
    model = BaseModel()
    model.eval()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)
    infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

    return infer_model
