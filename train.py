import numpy as np
import argparse
from pathlib import Path
from time import time

from models.dec import train_and_select_model
from data.utils import get_labeled_subset

from typing import Tuple
from numpy.typing import NDArray


def load_data(data_path: Path, shuffle: bool=True) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    X_train: NDArray = np.load(data_path / 'X_train.npy')
    y_train: NDArray = np.load(data_path / 'y_train.npy')
    if shuffle:
        p: NDArray = np.random.permutation(len(X_train))
        X_train, y_train = X_train[p], y_train[p]

    X_val: NDArray = np.load(data_path / 'X_val.npy')
    y_val: NDArray = np.load(data_path / 'y_val.npy')
    if shuffle:
        p: NDArray = np.random.permutation(len(X_val))
        X_val, y_val = X_val[p], y_val[p]

    return X_train, y_train, X_val, y_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/home/timodw/IDLab/unsupervised_gait_detection/processed_data')
    parser.add_argument('--results', default='/home/timodw/IDLab/unsupervised_gait_detection/results')
    parser.add_argument('--preprocessing', default='raw')
    parser.add_argument('--fold', default=0)
    parser.add_argument('--n_samples', default=5)
    parser.add_argument('--n_models', default=5)
    parser.add_argument('--n_iter', default=5)
    parser.add_argument('--dec_init', default='kmeans')
    parser.add_argument('--latent_dim', default=32)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # Create folder for storing the results and logs
    # results_folder = Path(args.results) / f"{args.preprocessing}" / f"fold_{args.fold}"
    # results_folder.mkdir(exist_ok=True, parents=True)

    # Loading the data for this fold
    data_folder = Path(args.data) / args.preprocessing / f"fold_{args.fold}"
    print(f"Loading data from '{data_folder}'...")
    ts = time()
    X_train, y_train, X_val, y_val = load_data(data_folder)
    n_clusters = len(np.unique(y_train))
    te = time()
    print(f"Loaded {X_train.shape[0]:,} training samples and {X_val.shape[0]:,} validation samples in {te - ts:.2f}s!\n")


    # Train the DEC models
    for i in range(args.n_iter):
        print(f"Training iteration {i + 1}/{args.n_iter}...")
        ts = time()
        X_train_selected, y_train_selected = get_labeled_subset(X_train, y_train, args.n_samples)
        train_and_select_model(X_train, X_train_selected, y_train_selected, args.n_models,
                               verbose=True, device=args.device)
        te = time()
        print(f"Training completed after {te - ts:.2f}s\n")
