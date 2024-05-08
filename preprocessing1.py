import glob
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from setting import CFG

from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def make_train_df():
    all_img_list = glob.glob('./data/train/*/*')
    all_img_list = [img_path.replace("\\", "/") for img_path in all_img_list]

    df = pd.DataFrame(columns=['img_path', 'label'])
    df['img_path'] = all_img_list
    df['label'] = df['img_path'].apply(lambda x : str(x).split('/')[3])

    train, val, _, _ = train_test_split(df, df['label'], test_size=0.3, stratify=df['label'], random_state=CFG['SEED'])

    le = preprocessing.LabelEncoder()
    train['label'] = le.fit_transform(train['label'])
    val['label'] = le.transform(val['label'])

    return train, val, le


def make_test_df():
    test = pd.read_csv('./data/test.csv')
    test['img_path'] = test['img_path'].str.replace("./","./data/")
    
    return test




class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        
        img_array = np.fromfile(img_path, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.img_path_list)




