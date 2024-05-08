from preprocessing1 import make_train_df, make_test_df
from make_loader import train_transform, test_transform, make_loader
import torch
from model import model_make
from submission1 import inference, submit_file

def main():
    device = torch.device('cpu') if torch.cuda.is_available() else torch.device('cpu')
    train2, val, le = make_train_df()
    test = make_test_df()
    train_loader, val_loader, test_loader =  make_loader(train2, val, test, train_transform, test_transform)
    best_model = model_make(train_loader, val_loader, device)
    preds = inference(le, best_model, test_loader, device)
    submit = submit_file(preds)

main()
