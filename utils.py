import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys, os, time, random, pickle
from tqdm import tqdm

from scipy.signal import find_peaks_cwt
from scipy.signal import butter, filtfilt

# Reproducibility
def seed_everything(SEED=2025):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)

# Predefined Variables
UNIT_TIME = 30
FS = 64

DEMO_CSV_FILENAME = 'dataset/participant_info_onehot.csv'
DATA_CSV_FOLDER   = 'dataset/raw'


# ====================
# Signal I/O
# ====================
def load_signal(filename):
    df = pd.read_csv(filename, index_col=0)
    df['TIMESTAMP'] = df['TIMESTAMP'] - df['TIMESTAMP'].iloc[0]
    return df


def slice_signal(df, time_init, duration=UNIT_TIME, fs=FS):
    idx_i = int(time_init * fs)
    idx_f = int((time_init + duration) * fs)
    
    return df.iloc[idx_i:idx_f].reset_index(drop=True)


def is_valid_epoch(df):
    for colname in df.columns[:-1]:
        if df[colname].isna().sum() > 0: return 0
    
    if -1 in df['Sleep_Stage'].unique(): return 0
    return 1


def get_label(df):
    return df['Sleep_Stage'].mode().values[0]


def time_init(df):
    return df['TIMESTAMP'].iloc[0]


def plot_signals(df, sig_list=['BVP'], axs=None):
    n_chs = len(sig_list)

    if axs is None:
        _, axs = plt.subplots(n_chs, 1, figsize=(15, 2 * n_chs))

    for i, sig in enumerate(sig_list):
        try:
            axs[i].plot(df['TIMESTAMP'] - time_init(df), df[sig], label=sig)
            axs[i].legend(loc='upper left')

            if 'ACC' in sig: axs[i].set_ylim(-75, +75)
            if 'HR' in sig:  axs[i].set_ylim(0, 180)
            if 'IBI' in sig: axs[i].set_ylim(0, 2)
            if 'TEMP' in sig: axs[i].set_ylim(30, 40)

        except:
            axs.plot(df['TIMESTAMP'] - time_init(df), df[sig], label=sig)
            axs.legend(loc='upper left')

            if 'ACC' in sig: axs.set_ylim(-100, +100)
            if 'HR' in sig:  axs.set_ylim(0, 180)
            if 'IBI' in sig: axs.set_ylim(0, 2)
            if 'TEMP' in sig: axs.set_ylim(30, 40)

    return axs


# ====================
# Frequency Filters
# ====================
def apply_bandpass_filter(signal, lowcut, highcut, fs=FS, order=4):
    b, a = butter(order, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band')
    filtered_signal = filtfilt(b, a, signal.values)

    return pd.Series(filtered_signal, index=signal.index)


def apply_lowpass_filter(signal, cutoff, fs=FS, order=4):
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    filtered_signal = filtfilt(b, a, signal.values)

    return pd.Series(filtered_signal, index=signal.index)


def apply_highpass_filter(signal, cutoff, fs=FS, order=4):
    b, a = butter(order, cutoff / (fs / 2), btype='high')
    filtered_signal = filtfilt(b, a, signal.values)

    return pd.Series(filtered_signal, index=signal.index)


# ====================
# PPG Features
# ====================
def get_ppg_peaks(signal, fs=FS):
    ppg_peaks = nk.ppg_findpeaks(signal, sampling_rate=fs)['PPG_Peaks']

    return ppg_peaks


def ppg_beat_features(ppg_peaks, result_dict:dict, fs=FS):
    ppg_peaks = np.array(ppg_peaks) / fs
    ibi_list = np.diff(ppg_peaks)
    hr_list = 60. / ibi_list

    # Heart Rate
    result_dict['PPG_HR_MEAN']      = np.mean(hr_list)
    result_dict['PPG_HR_STD']       = np.std(hr_list)
    result_dict['PPG_HR_MAX']       = np.max(hr_list)
    result_dict['PPG_HR_90P']       = np.percentile(hr_list, 90)
    result_dict['PPG_HR_10P']       = np.percentile(hr_list, 10)
    result_dict['PPG_HR_MIN']       = np.min(hr_list)
    result_dict['PPG_HR_VAR']       = (result_dict['PPG_HR_90P'] - result_dict['PPG_HR_10P']) / result_dict['PPG_HR_MEAN']

    # Shannon Entropy
    binwidth = 10
    bins = np.arange(0, 200 + binwidth, binwidth)
    hist, _ = np.histogram(hr_list, bins=bins)
    hist = hist[hist > 0] / len(hr_list)  # Remove zero entries
    result_dict['PPG_HR_ENTROPY'] = -np.sum(hist * np.log(hist))

    return result_dict


# ====================
# EDA Features
# ====================
def eda_features(signal, result_dict, fs=FS):
    result_dict['EDA_MEAN'] = np.mean(signal)
    result_dict['EDA_STD']  = np.std(signal)
    result_dict['EDA_MAX']  = np.max(signal)
    result_dict['EDA_MIN']  = np.min(signal)

    return result_dict


# ====================
# ACC Features
# ====================
def acc_mag_features(acc_mag, result_dict):
    result_dict['ACC_MAG_MEAN']    = np.mean(acc_mag)
    result_dict['ACC_MAG_STD']     = np.std(acc_mag)
    result_dict['ACC_MAG_DEV_MAX'] = np.max(acc_mag) - result_dict['ACC_MAG_MEAN']
    result_dict['ACC_MAG_DEV_90P'] = np.percentile(acc_mag, 90) - result_dict['ACC_MAG_MEAN']
    result_dict['ACC_MAG_DEV_10P'] = np.percentile(acc_mag, 10) - result_dict['ACC_MAG_MEAN']
    result_dict['ACC_MAG_DEV_MIN'] = np.min(acc_mag) - result_dict['ACC_MAG_MEAN']
    result_dict['ACC_MAG_RANGE']   = result_dict['ACC_MAG_DEV_MAX'] - result_dict['ACC_MAG_DEV_MIN']
    
    return result_dict


def acc_vec_features(ax, ay, az, result_dict):
    a_vec = np.c_[np.array(ax).ravel(), np.array(ay).ravel(), np.array(az).ravel()]
    a_mag = np.linalg.norm(a_vec, axis=1).reshape(-1, 1)

    # Deviation of ACC Vector (sum of a_delta)
    a_delta = a_vec[1:] - a_vec[:-1]
    a_delta = np.linalg.norm(a_delta, axis=1).reshape(-1, 1)
    result_dict['ACC_VEC_DEV_CUM'] = np.sum(a_delta)
    result_dict['ACC_VEC_DEV_CUM_LOG1P'] = np.log1p(result_dict['ACC_VEC_DEV_CUM'])

    # Deviation of ACC Direction (sum of angle)
    a_dot = np.sum(a_vec[1:] * a_vec[:-1], axis=1).reshape(-1, 1)
    a_mag_ = a_mag[1:] * a_mag[:-1]
    a_dot /= a_mag_
    a_dot = np.clip(a_dot, -1, 1)
    result_dict['ACC_ANG_DEV_CUM'] = np.sum(np.arccos(a_dot))
    result_dict['ACC_ANG_DEV_CUM_LOG1P'] = np.log1p(result_dict['ACC_ANG_DEV_CUM'])
    
    return result_dict


# ====================
# HR, IBI, TEMP Features
# ====================
def hr_features(signal, result_dict, fs=FS):
    result_dict['HR_MEAN'] = np.mean(signal)
    result_dict['HR_STD']  = np.std(signal)
    result_dict['HR_MAX']  = np.max(signal)
    result_dict['HR_MIN']  = np.min(signal)
    
    return result_dict


def ibi_features(signal, result_dict, fs=FS):
    result_dict['IBI_MEAN'] = np.mean(signal)
    result_dict['IBI_STD']  = np.std(signal)
    result_dict['IBI_MAX']  = np.max(signal)
    result_dict['IBI_MIN']  = np.min(signal)

    return result_dict


def temp_features(signal, result_dict, fs=FS):
    result_dict['TEMP_MEAN'] = np.mean(signal)
    result_dict['TEMP_STD']  = np.std(signal)
    result_dict['TEMP_MAX']  = np.max(signal)
    result_dict['TEMP_MIN']  = np.min(signal)

    return result_dict


# ====================
# Full Pipeline
# ====================
def preprocess_epoch(df_slice, fs=FS):
    result_dict = {}

    result_dict['EpochValidity'] = is_valid_epoch(df_slice)
    if result_dict['EpochValidity'] == False: return result_dict, None
    
    result_dict['EpochLabel'] = get_label(df_slice)

    # BVP Preprocessing
    df_slice['BVP_PREP'] = apply_bandpass_filter(df_slice['BVP'], lowcut=0.5, highcut=5, order=3, fs=fs)
    ppg_peaks = get_ppg_peaks(df_slice['BVP_PREP'])
    result_dict = ppg_beat_features(ppg_peaks, result_dict)

    # EDA Preprocessing
    df_slice['EDA_PREP'] = apply_lowpass_filter(df_slice['EDA'], cutoff=0.5, order=3, fs=fs)
    result_dict = eda_features(df_slice['EDA_PREP'], result_dict)

    # ACC Preprocessing
    df_slice['ACC_X_PREP'] = apply_lowpass_filter(df_slice['ACC_X'], cutoff=5, order=3, fs=fs)
    df_slice['ACC_Y_PREP'] = apply_lowpass_filter(df_slice['ACC_Y'], cutoff=5, order=3, fs=fs)
    df_slice['ACC_Z_PREP'] = apply_lowpass_filter(df_slice['ACC_Z'], cutoff=5, order=3, fs=fs)

    df_slice['ACC_MAG_PREP']= np.sqrt(df_slice['ACC_X_PREP']**2 + df_slice['ACC_Y_PREP']**2 + df_slice['ACC_Z_PREP']**2)
    result_dict = acc_mag_features(df_slice['ACC_MAG_PREP'], result_dict)
    result_dict = acc_vec_features(df_slice['ACC_X_PREP'], df_slice['ACC_Y_PREP'], df_slice['ACC_Z_PREP'], result_dict)

    # HR & IBI & TEMP Preprocessing
    df_slice['HR_PREP'] = apply_lowpass_filter(df_slice['HR'], cutoff=1, order=3, fs=fs)
    df_slice['IBI_PREP'] = apply_lowpass_filter(df_slice['IBI'], cutoff=1, order=3, fs=fs)
    df_slice['TEMP_PREP'] = apply_lowpass_filter(df_slice['TEMP'], cutoff=1, order=3, fs=fs)

    result_dict = hr_features(df_slice['HR_PREP'], result_dict)
    result_dict = ibi_features(df_slice['IBI_PREP'], result_dict)
    result_dict = temp_features(df_slice['TEMP_PREP'], result_dict)

    # Columns
    df_slice = df_slice[['TIMESTAMP', 'Sleep_Stage', 'BVP_PREP', 'EDA_PREP', 'ACC_X_PREP', 'ACC_Y_PREP', 'ACC_Z_PREP', 'ACC_MAG_PREP', 'HR_PREP', 'IBI_PREP', 'TEMP_PREP']]
    
    return result_dict, df_slice