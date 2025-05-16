from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from utils import moving_average_smoothing, standardize_time_series
from utils import normalize_time_series, fft_based_filtering, peak_based_segmentation
from utils import sliding_window, resample_time_series, fft_time_series
from utils import balance_dataset, pca_windows

from typing import Tuple, List, Dict, Union, Any
from numpy.typing import NDArray

N_FOLDS = 4
DATA_PATH = Path('/data/IDLab/horse_data/HorsingAround/data/csv')
PROCESSED_DATA_PATH = Path('../processed_data') 

def process_data(df, movements_of_interest,
                 standardized=False, normalized=False,
                 peak_segmented=False,
                 smoothed=False,
                 resampled=False, resampled_sr=50,
                 dft_filtered=False, dft_filtered_hz=10, dft=False,
                 sampling_rate=100, window_length=2):
    label_mapping = {m: i for i, m in enumerate(movements_of_interest)}
    window_size = window_length * sampling_rate
    step_size = window_size
    X, y = df['norm'].values, df.label.values
        
    if smoothed:
        X = moving_average_smoothing(X, window_size=50)
    if standardized:
        X = standardize_time_series(X)
    if normalized:
        X = normalize_time_series(X, symmetric=False)
    if dft_filtered:
        X = fft_based_filtering(X, dft_filtered_hz, 100)
    
    if peak_segmented:
        X_windows, y_windows = peak_based_segmentation(
            X, y, window_size=window_size, step_size=step_size, peak_distance=100
        )
    else:
        X_windows, y_windows = sliding_window(
            X, y, window_size=window_size, step_size=step_size
        )
    
    selected_windows = np.isin(y_windows, movements_of_interest)
    X_windows = X_windows[selected_windows]
    y_windows = y_windows[selected_windows]
    y_windows = np.asarray([label_mapping[l] for l in y_windows])
    
    if len(X_windows) > 0:
        if resampled:
            resampled_windows = []
            for window in X_windows:
                resampled_windows.append(resample_time_series(window, resampled_sr, 100))
            X_windows = np.stack(resampled_windows)
        if dft:
            transformed_windows = []
            for window in X_windows:
                transformed_windows.append(fft_time_series(window, 100))
            X_windows = np.stack(transformed_windows)           
        
    return X_windows, y_windows


def get_activity_distribution(root_folder: Path, movements_of_interest: List[str]) -> pd.DataFrame:
    activity_distribution = pd.read_csv(root_folder / 'activity_distribution.csv')
    columns_of_interest: List[str] = []
    for movement in movements_of_interest:
        for column in activity_distribution.columns:
            if column.startswith(movement):
                columns_of_interest.append(column)
    activity_distribution = activity_distribution[['Row'] + columns_of_interest]
    return activity_distribution


def get_horses_of_interest(activity_distribution: pd.DataFrame, movements_of_interest: List[str]) -> List[str]:
    horses_of_interest: List[str] = []
    for i, row in activity_distribution.iterrows():
        if row['Row'] != 'total':
            movement_counts: Dict[str, float] = defaultdict(float)
            for movement in activity_distribution.columns[1:]:
                movement_type = movement.split('_')[0]
                count: float = row[movement]
                if count == count:
                    movement_counts[movement_type] += row[movement]
            valid = True
            for movement in movements_of_interest:
                if movement_counts[movement] < 1.:
                    valid = False
                    break
            if valid:
                horses_of_interest.append(row['Row'])
    return horses_of_interest


def get_data_for_horse(dataframes: List[pd.DataFrame],
                       movements_of_interest: List[str],
                       methods: Dict[str, Any]) -> Tuple[NDArray, NDArray]:
    X_total, y_total = None, None
    for df in dataframes:
        X, y = process_data(df, movements_of_interest, **methods)
        if len(X) == 0:
            continue
        if X_total is not None:
            X_total = np.concatenate([X_total, X])
            y_total = np.concatenate([y_total, y])
        else:
            X_total = X
            y_total = y
    return X_total, y_total


def get_horse_dataframes(root_folder: Path, horses_of_interest: List[str]) -> Dict[str, List[pd.DataFrame]]:
    horses_dataframes = defaultdict(list)
    for horse in horses_of_interest:
        print(f'Loading horse {horse}')
        for f in root_folder.glob(f'*{horse}*'):
            df = pd.read_csv(f, low_memory=False)
            df = df[['label', 'segment', 'Ax', 'Ay', 'Az']].dropna()
            df['norm'] = np.sqrt(df['Ax']**2 + df['Ay']**2 + df['Az']**2)
            if len(df) > 0:
                horses_dataframes[horse].append(df)
    return horses_dataframes


if __name__ == '__main__':
    root_folder = Path('/data/IDLab/horse_data/HorsingAround/data/csv')
    movements_of_interest = ['standing', 'walking', 'trotting', 'galloping']
    preprocessing_methods = ['raw', 'standardized', 'normalized', 'peak_segmented', 'smoothed',
                             'resampled_25', 'resampled_50', 'pca', 'dft_filter', 'dft']

    activity_distribution = get_activity_distribution(root_folder, movements_of_interest)
    horses_of_interest = get_horses_of_interest(activity_distribution, movements_of_interest)
    print('Loading horses!')
    horse_dataframes = get_horse_dataframes(root_folder, horses_of_interest)
    for method in preprocessing_methods:
        for i, (training_horses, validation_horses) in enumerate(KFold(n_splits=N_FOLDS).split(horses_of_interest)):
            print(f"Generating data for {method}, fold {i + 1}")
            output_folder = PROCESSED_DATA_PATH / f"{method}/fold_{i}"
            output_folder.mkdir(exist_ok=True, parents=True)

            preprocessing_methods: Dict[str, Union[bool, int]] = dict()
            if method == 'standardized':
                preprocessing_methods['standardized'] = True
            elif method == 'normalized':
                preprocessing_methods['normalized'] = True
            elif method == 'peak_segmented':
                preprocessing_methods['peak_segmented'] = True
            elif method == 'smoothed':
                preprocessing_methods['smoothed'] = True
            elif method.startswith('resampled'):
                _, hz = method.split('_')
                hz = int(hz)
                preprocessing_methods['resampled'] = True
                preprocessing_methods['resampled_sr'] = hz
            elif method == 'dft_filter':
                preprocessing_methods['dft_filtered'] = True
                preprocessing_methods['dft_filtered_hz'] = 10
            elif method == 'dft':
                preprocessing_methods['dft'] = True
            
            X_train, y_train = None, None
            for horse_id in training_horses:
                horse_id = horses_of_interest[horse_id]
                X, y = get_data_for_horse(horse_dataframes[horse_id], movements_of_interest,
                                          preprocessing_methods)
                if X_train is not None:
                    X_train = np.concatenate([X_train, X])
                    y_train = np.concatenate([y_train, y])
                else:
                    X_train = X
                    y_train = y

            X_validation, y_validation = None, None
            for horse_id in validation_horses:
                horse_id = horses_of_interest[horse_id]
                X, y = get_data_for_horse(horse_dataframes[horse_id], movements_of_interest,
                                          preprocessing_methods)
                if X_validation is not None:
                    X_validation = np.concatenate([X_validation, X])
                    y_validation = np.concatenate([y_validation, y])
                else:
                    X_validation = X
                    y_validation = y
            X_train, y_train = balance_dataset(X_train, y_train)
            if method == 'pca':
                X_train, pca = pca_windows(X_train, n_components=25)
                X_validation, _ = pca_windows(X_validation, n_components=25, pca=pca)

            np.save(output_folder / 'X_train.npy', X_train)
            np.save(output_folder / 'y_train.npy', y_train)
            np.save(output_folder / 'X_val.npy', X_validation)
            np.save(output_folder / 'y_val.npy', y_validation)