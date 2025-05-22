from sklearn.mixture import GaussianMixture


from typing import Optional, Any, Dict
from numpy.typing import NDArray


def train_gmm(X_train: NDArray, n_clusters: int=4, gmm_args: Optional[Dict[str, Any]]=None, **kwargs) -> GaussianMixture:
    if gmm_args is None:
        gmm_args = {}
    gmm = GaussianMixture(n_clusters, **gmm_args)
    gmm.fit(X_train)
    return gmm