import numpy as np
import argparse
from pathlib import Path
from time import time
import pickle as pkl
import csv

from models.utils import train_and_select_model, evaluate_selected_model
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
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument('--n_samples', default=5, type=int)
    parser.add_argument('--n_models', default=5, type=int)
    parser.add_argument('--n_iter', default=5, type=int)
    parser.add_argument('--latent_dim', default=32, type=int)
    parser.add_argument('--balanced', default=False, action='store_true')
    parser.add_argument('--model_type', default='dec_kmeans')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # Create folder for storing the results and logs
    results_folder = Path(args.results) / f"{args.model_type}" / ('balanced' if args.balanced else 'unbalanced') / f"{args.preprocessing}" / f"fold_{args.fold}"
    results_folder.mkdir(exist_ok=True, parents=True)

    # Loading the data for this fold
    data_folder = Path(args.data) / ('balanced' if args.balanced else 'unbalanced') / args.preprocessing / f"fold_{args.fold}"
    print(f"Loading data from '{data_folder}'...")
    ts = time()
    X_train, y_train, X_val, y_val = load_data(data_folder)
    n_clusters = len(np.unique(y_train))
    te = time()
    print(f"Loaded {X_train.shape[0]:,} training samples and {X_val.shape[0]:,} validation samples in {te - ts:.2f}s!\n")

    if args.model_type.startswith('dec'):
        init_method = args.model_type.split('_')[1]
        model_parameters = {'dec_clustering_init': init_method}
        args.model_type = 'dec'
    else:
        model_parameters = {}

    # Train the models
    results = []
    for i in range(args.n_iter):
        print(f"Iteration {i + 1}/{args.n_iter}...")
        ts = time()
        X_train_selected, y_train_selected = get_labeled_subset(X_train, y_train, args.n_samples)
        model, label_mapping = train_and_select_model(X_train, X_train_selected, y_train_selected,
                                                      args.n_models, args.model_type, model_parameters,
                                                      verbose=True, device=args.device)
        if args.model_type == 'dec':
            pass
        else:
            pkl.dump(model, open(results_folder / f"model_{i}.pkl", 'wb'))
            pkl.dump(label_mapping, open(results_folder / f"label_mapping_{i}.pkl", 'wb'))
        precision, recall, f1_score, _ = evaluate_selected_model(X_val, y_val, model, label_mapping, device=args.device)
        results.append([precision, recall, f1_score])
        print(f"Precision: {precision:.2%}; Recall: {recall:.2%}; F1-Score: {f1_score:.2%}")
        te = time()
        time_str = f"Iteration completed after {te - ts:.2f}s"
        print(time_str, end='\n\n')
        print(f"{'#' * len(time_str)}\n")

    results = np.asarray(results)
    np.save(results_folder / 'results.npy', results)
    print(f"Precision:\t{np.mean(results[:, 0]):.2%}±{100 * np.std(results[:, 0]):.2f}")
    print(f"Recall:\t\t{np.mean(results[:, 1]):.2%}±{100 * np.std(results[:, 1]):.2f}")
    print(f"F1-Score:\t{np.mean(results[:, 2]):.2%}±{100 * np.std(results[:, 2]):.2f}")
