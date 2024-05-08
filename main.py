import warnings
warnings.filterwarnings(action='ignore')

import torch
import pandas as pd

from preprocessing import train_loader, val_loader, test_loader, CFG
from model import BaseModel, train, inference, device

model = BaseModel()

model.eval()
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)
infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

torch.save(infer_model.state_dict(), "best_model4.pth")

new_model = BaseModel()
new_model.load_state_dict(torch.load("best_model4.pth"))

preds = inference(infer_model, test_loader, device)

submit = pd.read_csv('./data/sample_submission.csv')
submit['label'] = preds
submit.to_csv('./baseline_submit07.csv', index=False)