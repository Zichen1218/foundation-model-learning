import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pathlib import Path
import torch
from torch.utils.data import Dataset



def preprocess(df_train, df_test):
    """
    Preprocess raw housing dataframes into model-ready numpy arrays.

    Returns a dict: {'X_train', 'y_train', 'X_test', 'feature_names'}
    """
    # YOUR CODE HERE
    y_train = np.log1p(df_train['SalePrice'].values).astype('float32')
    df_train = df_train.drop(columns=['Id','SalePrice'])
    df_test = df_test.drop(columns=['Id'])
    split_idx = len(df_train)
    df_total = pd.concat([df_train,df_test]).reset_index(drop=True)

    cat_col = df_total.select_dtypes(include='object').columns
    df_total[cat_col] = df_total[cat_col].fillna('Missing')
    df_total = pd.get_dummies(df_total,columns=cat_col)

    num_col = df_total.select_dtypes(include='number').columns
    imputer = SimpleImputer(strategy='median')
    df_total[num_col] = imputer.fit_transform(df_total[num_col])
    scaler = StandardScaler()
    df_total[num_col] = scaler.fit_transform(df_total[num_col])
    feature_names = df_total.columns
    X_train = df_total.iloc[:split_idx].values.astype('float32')
    X_test = df_total.iloc[split_idx:].values.astype('float32')
    return {
        'X_train':X_train,
        'y_train':y_train,
        'X_test':X_test,
        'feature_names':feature_names,
    }
class HousingDataset(Dataset):
    """
    PyTorch Dataset for tabular housing data.

    Args:
        X (np.ndarray): features, shape [N, D], float32
        y (np.ndarray or None): targets, shape [N], float32
    """
    def __init__(self, X, y=None):
        # YOUR CODE HERE
        self.X = torch.tensor(X)
        if y is not None:
            self.y = torch.tensor(y)
            

    def __len__(self):
        # YOUR CODE HERE
        return len(self.X)

    def __getitem__(self, idx):
        # YOUR CODE HERE
        if self.y is not None:
            return self.X[idx],self.y[idx]
        return self.X[idx]



DATA_DIR = Path('house-prices')

df_train = pd.read_csv(DATA_DIR / 'train.csv')
df_test  = pd.read_csv(DATA_DIR / 'test.csv')

data = preprocess(df_train.copy(), df_test.copy())

X_train = data['X_train']
y_train = data['y_train']
X_test  = data['X_test']