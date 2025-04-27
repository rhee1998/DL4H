import numpy as np
import pandas as pd
import sys, os

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

# Run the Full Model
os.system(f'python3 data_preprocess.py --sid={SID}')
os.system(f'python3 run_epoch_clf.py --sid={SID} --clinical_info={CI}')
os.system(f'python3 run_post_lstm.py --sid={SID} --clinical_info={CI} --window_size={WINDOW_SIZE}')