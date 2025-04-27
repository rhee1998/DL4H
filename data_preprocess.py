import numpy as np
import pandas as pd
import os, random
from tqdm import tqdm

from utils import *


# Reproducibility
def seed_everything(SEED=2025):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)


# Predefined Variables
UNIT_TIME = 30
FS = 64
MAX_WINDOW = 20

DEMO_CSV_FILENAME = 'dataset/clinical_info_onehot.csv'
DATA_RAW_FOLDER   = 'dataset/raw'
DATA_PREP_FOLDER  = 'dataset/preproc'

# Parse Command Line Arguments
args_dict = {}
for argv in sys.argv[1:]:
    if not (argv.startswith('--') and (argv.count('=') == 1)):
        break
    key = argv.split('=')[0][2:]
    val = argv.split('=')[1]
    args_dict[key] = val

sid = args_dict.get('sid', None)
if sid is None:
    print('Usage: python3 data_preprocess.py --sid=<SID>')
    sys.exit(1)


# Load Clinical Information
df_demo = pd.read_csv(DEMO_CSV_FILENAME)


# Preprocess Full-Length Signals
# Generate 30-sec Epochs
os.makedirs(f'{DATA_PREP_FOLDER}/{sid}', exist_ok=True)
df = load_signal(f'{DATA_RAW_FOLDER}/{sid}.csv')
df_res = pd.DataFrame()
os.makedirs(f'{DATA_PREP_FOLDER}/{sid}', exist_ok=True)

for time_init_ in tqdm(range(0, int(df['TIMESTAMP'].max()) - UNIT_TIME, UNIT_TIME)):
    df_slice = slice_signal(df, time_init=time_init_)
    result_dict, df_slice = preprocess_epoch(df_slice)

    df_res = pd.concat([df_res, pd.DataFrame(result_dict, index=[time_init_])])

    if df_slice is not None:
        df_slice.to_csv(f'{DATA_PREP_FOLDER}/{sid}/{time_init_:05d}.csv', index=False)

df_res['IncludeStudy'] = 1
for idx in range(MAX_WINDOW):
    df_res.loc[idx * UNIT_TIME, 'IncludeStudy'] = 0

for idx in df_res.index:
    if df_res.loc[idx, 'EpochValidity'] == 1:
        continue

    for idx_ in range(MAX_WINDOW + 1):
        if idx + idx_ * UNIT_TIME > df_res.index.max():
            break

        df_res.loc[idx + idx_ * UNIT_TIME, 'IncludeStudy'] = 0
    
df_res.to_csv(f'{DATA_PREP_FOLDER}/{sid}/summary.csv', index=True)