from sklearn.cluster import KMeans

from typing import Optional, Any, Dict
from numpy.typing import NDArray


def train_kmeans(X_train: NDArray, n_clusters: int=4, kmeans_args: Optional[Dict[str, Any]]=None, **kwargs) -> KMeans:
    if kmeans_args is None:
        kmeans_args = {}
    kmeans = KMeans(n_clusters, **kmeans_args)
    kmeans.fit(X_train)
    return kmeans