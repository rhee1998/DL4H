import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys, os, time, random, warnings, pickle
from tqdm import tqdm

from sklearn.metrics import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# Predefined Constants
N_PREFETCH = 4
N_WORKERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

N_FOLD              = 5
POSTPROC_DIR        = 'dataset/postproc'


# Parse Command Line Arguments
args_dict = {}
for argv in sys.argv[1:]:
    if not (argv.startswith('--') and (argv.count('=') == 1)):
        break
    key = argv.split('=')[0][2:]
    val = argv.split('=')[1]
    args_dict[key] = val

SID = args_dict.get('sid', None)
WINDOW_SIZE = int(args_dict.get('window_size', -1))
CI = args_dict.get('clinical_info', None)

if (SID is None) or (WINDOW_SIZE not in [5, 10, 15, 20]) or (CI not in ['True', 'False']):
    print('Usage: python3 run_post_lstm.py --sid=<SID> --clinical_info=<True/False> --window_size=<WINDOW_SIZE>')
    sys.exit(1)

CI = {'True':True, 'False':False}[CI]


# Torch Dataset Class
class PostProcDataset(Dataset):
    def __init__(self, df, window_size=WINDOW_SIZE):
        self.window_size = window_size
        columns = [str(WINDOW_SIZE - i - 1) for i in range(window_size)] + ['EpochClfProba']
        
        self.input = df[columns].to_numpy()
        self.target = df['EpochLabel'].to_numpy()

    def __len__(self):
        return self.input.shape[0]
    
    def __getitem__(self, idx):
        x = self.input[idx]
        x = torch.tensor(x, dtype=torch.float32)
        x = x.unsqueeze(1)
        
        y = self.target[idx]
        y = torch.tensor([1 - y, y], dtype=torch.float32)
        
        return x, y
    

# Define LSTM Model
class PostProcLSTM(nn.Module):
    def __init__(self, input_size=WINDOW_SIZE, hidden_size=8, n_layers=1, dropout=0.2):
        super(PostProcLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = self.fc(x)
        return x
    

# Test Model
def evaluate_model(valid_loader, model, device=DEVICE):
    model.to(device)
    model.eval()
    
    y_true, y_pred_proba = None, None
    n_batch = len(valid_loader)
    pbar = tqdm(enumerate(valid_loader), total=n_batch)

    with torch.no_grad():
        for i_batch, (X_batch, y_true_batch) in pbar:
            X_batch, y_true_batch = X_batch.to(device), y_true_batch.to(device)
            
            y_pred_batch = F.sigmoid(model(X_batch))
            y_pred_batch = y_pred_batch.detach().cpu().numpy()[:, 1].reshape((-1, 1))
            y_true_batch = y_true_batch.detach().cpu().numpy()[:, 1].reshape((-1, 1))

            try:
                y_pred_proba = np.r_[y_pred_proba, y_pred_batch]
                y_true = np.r_[y_true, y_true_batch]
            except:
                y_pred_proba = y_pred_batch
                y_true = y_true_batch
                
            pbar.set_description(f'Evaluating ... {1 + i_batch}/{n_batch}')
    
    return y_true, y_pred_proba


'''
MAIN FUNCTION
'''
df = pd.read_csv(f'{POSTPROC_DIR}/{SID}/epoch_clf_results_CI={CI}.csv', index_col=0)
ds = PostProcDataset(df, window_size=WINDOW_SIZE)
data_loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=N_WORKERS, prefetch_factor=N_PREFETCH)

model_path = f'model/post_LSTM_WIN={WINDOW_SIZE}_CI={CI}'
hparams = pickle.load(open(f'{model_path}/hparams.pkl', 'rb'))
hidden_size, n_layers = hparams['hidden_size'], hparams['n_layers']
model = PostProcLSTM(input_size=WINDOW_SIZE+1, hidden_size=hidden_size, n_layers=n_layers)

for i_fold in range(N_FOLD):
    model.load_state_dict(torch.load(f'{model_path}/weights_{i_fold}.pth'))
    _, y_pred_proba = evaluate_model(data_loader, model)

    df[f'fold_{i_fold}'] = y_pred_proba
df['PostProcProba'] = df[[f'fold_{i_fold}' for i_fold in range(N_FOLD)]].mean(axis=1)
df = df[['EpochLabel', 'EpochClfProba', 'PostProcProba']]

os.makedirs(f'results/{SID}', exist_ok=True)
df.to_csv(f'results/{SID}/final_results_WIN={WINDOW_SIZE}_CI={CI}.csv')