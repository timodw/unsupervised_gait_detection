import numpy as np
from sklearn.decomposition import PCA

import scipy
from scipy.signal import find_peaks
from scipy.fft import fft, ifft, fftfreq


def balance_dataset(X, y):
    labels, label_counts = np.unique(y, return_counts=True)
    minority_label, minority_count = labels[np.argmin(label_counts)], np.min(label_counts)
    X_sel, y_sel = [], []
    for i, label in enumerate(labels):
        if label != minority_label:
            selected_samples = np.random.choice(label_counts[i], minority_count, replace=False)
            X_sel.append(X[y == label][selected_samples])
            y_sel.append(y[y == label][selected_samples])
        else:
            X_sel.append(X[y == label])
            y_sel.append(y[y == label])
    return np.concatenate(X_sel), np.concatenate(y_sel)


def pca_windows(X, n_components, pca=None):
    if pca is None:
        pca = PCA(n_components=n_components)
        pca.fit(X)
    return pca.transform(X), pca


def moving_average_smoothing(X, window_size=20):
    X = X.copy()
    return np.convolve(X, np.ones(window_size), 'valid') / window_size


def standardize_time_series(X):
    x_mean, x_std = X.mean(), X.std()
    X = X.copy()
    X -= x_mean
    X /= x_std
    return X


def normalize_time_series(X, symmetric=False):
    x_min, x_max = X.min(), X.max()
    X = X.copy()
    X -= x_min
    X /= (x_max - x_min)
    if symmetric:
        X *= 2
        X -= 1
    return X


def fft_based_filtering(X, high_cutoff, sampling_rate):
    frequencies = fftfreq(len(X), 1 / sampling_rate)
    mask = np.abs(frequencies) > high_cutoff
    
    X_fft = fft(X)
    X_fft[mask] = 0.
    X = np.abs(ifft(X_fft))
    
    return X


def peak_based_segmentation(X, y, window_size, step_size, peak_distance=10):
    X_windows = []
    y_windows = []

    for i in range(step_size, len(X) - window_size, step_size):
        #import pdb; pdb.set_trace()
        peaks = find_peaks(X[i - step_size:i + step_size], distance=peak_distance)[0]
        selected_peak = np.argmin(np.abs(peaks - step_size))
        window_start = i - step_size + selected_peak
        window = X[window_start:window_start + window_size]
        window_labels = y[window_start:window_start + window_size]
        y_unique = np.unique(window_labels)
        if len(y_unique) == 1:
            X_windows.append(window)
            y_windows.append(y_unique[0].split('-')[0])

    X_windows = np.stack(X_windows)
    y_windows = np.asarray(y_windows)
    
    return X_windows, y_windows


def sliding_window(X, y, window_size, step_size):
    X_windows = []
    y_windows = []

    for start_i in range(0, len(X) - window_size, step_size):
        y_unique = np.unique(y[start_i:start_i + window_size])
        if len(y_unique) == 1:
            X_windows.append(X[start_i:start_i + window_size])
            y_windows.append(y_unique[0].split('-')[0])

    X_windows = np.stack(X_windows)
    y_windows = np.asarray(y_windows)
    
    return X_windows, y_windows


def resample_time_series(X, new_sampling_rate, original_sampling_rate):
    ratio = new_sampling_rate / original_sampling_rate
    original_size = len(X)
    new_size = int(original_size * ratio)
    x_coord_original = np.arange(0, original_size)
    x_coord_new = np.linspace(0, original_size, new_size)
    X_resampled = np.interp(x_coord_new, x_coord_original, X)
    return X_resampled


def fft_time_series(X, sampling_rate):
    X_fft = fft(X)
    X_fft[0] = 0.
    X_fft_freq = fftfreq(len(X), 1 / sampling_rate)
    return np.abs(X_fft)[:len(X) // 2]