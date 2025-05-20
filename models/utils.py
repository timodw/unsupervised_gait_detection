import numpy as np
from time import time
import torch

from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import accuracy_score

from scipy.optimize import linear_sum_assignment

from models.dec import train_dec_end_to_end

from typing import Optional, Dict, Any
from numpy.typing import NDArray


def train_and_select_model(X_train_full: NDArray, X_train_labeled: NDArray, y_train: NDArray,
                           n_models: int, model_type='kmeans', model_parameters: Optional[Dict[str, Any]]=None,
                           verbose=True, device='cpu') -> Any:
    if model_parameters is None:
        model_parameters = {}
    if verbose:
        print(f"Training {n_models} {model_type} models...")
        ts = time()
    if model_type == 'dec':
        train_fn = train_dec_end_to_end
    models = [train_fn(X_train_full, verbose=False, device=device, **model_parameters) for i in range(n_models)]
    if verbose:
        te = time()
        print(f"Training took {te - ts:.2f}s!\n")
        print(f"Selecting best performing model...")
        ts = time()

    max_acc = .0
    max_acc_i = -1
    if model_type == 'dec':
        X_train_labeled = torch.from_numpy(X_train_labeled).to(torch.float32).to(device)
    for i, model in enumerate(models):
        y_pred = model(X_train_labeled)
        if model_type == 'dec':
            y_pred = np.argmax(y_pred.detach().cpu().numpy(), axis=1)
        label_mapping = get_label_mapping(y_train, y_pred)
        y_pred = np.array([label_mapping[l] for l in y_pred])
        acc = accuracy_score(y_train, y_pred)
        if acc > max_acc:
            max_acc = acc
            max_acc_i = i
        if verbose:
            print(f"Model {i + 1}/{n_models} scored {acc:.2%};")
    
    if verbose:
        te = time()
        print(f"Selected model {max_acc_i + 1} with an accuracy of {max_acc:.2%} in {te - ts:.2f}s!\n")


def get_label_mapping(y_true: NDArray, y_pred: NDArray) -> Dict[int, int]:
    labels = np.unique(y_true)
    cm = contingency_matrix(y_true, y_pred)
    true_labels, cluster_labels = linear_sum_assignment(cm, maximize=True)
    label_mapping = {cluster_l: true_l for cluster_l, true_l in zip(cluster_labels, true_labels)}
    if len(label_mapping) != len(labels):
        # Not all clusters are part of y_pred, we need to manually assign a label to them
        unassigned_clusters = list(set(labels) - label_mapping.keys())
        for l in labels:
            if l not in label_mapping:
                label_mapping[l] = unassigned_clusters[0]
                unassigned_clusters = unassigned_clusters[1:]

    return label_mapping