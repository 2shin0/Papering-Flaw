import torch
import tqdm
import pandas as pd

def inference(le, model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)
            
            pred = model(imgs)
            
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
    
    preds = le.inverse_transform(preds)
    return preds
    
def submit_file(preds):
    submit = pd.read_csv('./data/sample_submission.csv')
    submit['label'] = preds
    submit.to_csv('./baseline_submit07.csv', index=False)
    return submit
