import numpy as np
import pandas as pd
import sys, os, warnings, pickle
from tqdm import tqdm

warnings.filterwarnings('ignore')

# File Paths
N_FOLD              = 5
WINDOW_SIZE         = 20
PREPROC_DIR         = 'dataset/preproc'
DEMO_FILENAME       = 'dataset/clinical_info_onehot.csv'

# Parse Command Line Arguments
args_dict = {}
for argv in sys.argv[1:]:
    if not (argv.startswith('--') and (argv.count('=') == 1)):
        break
    key = argv.split('=')[0][2:]
    val = argv.split('=')[1]
    args_dict[key] = val

CI = args_dict.get('clinical_info', None)
SID = args_dict.get('sid', None)

if (not CI in ['True', 'False']) or (SID is None):
    print('Usage: python3 run_epoch_clf.py --sid=<SID> --clinical_info=<True/False>')
    sys.exit(1)

CI = {'True':True, 'False':False}[CI]

# Load Models
SCALER_FILENAME     = f'model/scaler_CI={CI}.pkl'
EPOCH_CLF_FILENAME  = f'model/epoch_clf_CI={CI}.pkl'


'''
A Few Utility Functions
'''
def run_epoch_clf_xgb(df, epoch_clf, n_fold=N_FOLD):
    y_pred_proba = np.zeros((n_fold, len(df)))
    inp_features = df.columns[1:-1]
    for i in range(n_fold):
        y_pred_proba[i] = epoch_clf[f'fold_{i}'].predict_proba(df[inp_features])[:, 1]

    y_pred_proba = np.mean(y_pred_proba, axis=0)
    return y_pred_proba

def load_epoch_clf_results(sid, scaler, epoch_clf, CI=CI, n_fold=N_FOLD, window_size=WINDOW_SIZE):
    # Load Preprocessed Data
    df = pd.read_csv(f'{PREPROC_DIR}/{sid}/summary.csv', index_col=0)
    df['SID'] = sid

    # Merge Demographic Information (prn)
    if CI:
        df_demo = pd.read_csv(DEMO_FILENAME, index_col=0)
        df = pd.merge(df, df_demo, left_on='SID', right_index=True)

    # Drop Unnecessary Columns
    df.set_index('SID', inplace=True)
    df = df[df['EpochValidity'] == 1].drop(columns=['EpochValidity'])

    # Scale Input Features
    inp_features = df.columns[1:-1]
    df[inp_features] = scaler.transform(df[inp_features])

    # Run Epoch Classifier
    y_pred_proba = np.zeros((n_fold, len(df)))
    for i in range(n_fold):
        y_pred_proba[i] = epoch_clf[f'fold_{i}'].predict_proba(df[inp_features])[:, 1]

    y_pred_proba = np.mean(y_pred_proba, axis=0)
    df['EpochClfProba'] = y_pred_proba
    df = df[['EpochLabel', 'IncludeStudy', 'EpochClfProba']]

    # Shift EpochClfProba
    df_tmp = np.zeros((len(df), window_size))
    for i in range(window_size):
        df_tmp[:, -i-1] = df['EpochClfProba'].shift(i + 1)

    df_tmp = pd.DataFrame(df_tmp, columns=[str(i) for i in range(window_size)], index=df.index)
    df = pd.concat([df, df_tmp], axis=1)

    # Filter & Drop Unnecessary Columns
    df = df[df['IncludeStudy'] == 1].drop(columns=['IncludeStudy'])

    return df

'''
Main Function
'''
df_demo = pd.read_csv(DEMO_FILENAME)
scaler = pickle.load(open(SCALER_FILENAME, 'rb'))
epoch_clf = pickle.load(open(EPOCH_CLF_FILENAME, 'rb'))

df = load_epoch_clf_results(SID, scaler, epoch_clf, CI=CI, n_fold=N_FOLD, window_size=WINDOW_SIZE)
    
os.makedirs(f'dataset/postproc/{SID}', exist_ok=True)
df.to_csv(f'dataset/postproc/{SID}/epoch_clf_results_CI={CI}.csv', encoding='utf-8-sig')