import torch
from time import time
import numpy as np
from time import time

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import accuracy_score

from scipy.optimize import linear_sum_assignment

from numpy.typing import NDArray
from typing import Optional, Dict, Any


class DenseEncoder(torch.nn.Module):
    
    def __init__(self, input_dim, latent_dim,
                 layer_sizes=[512, 128]):
        super(DenseEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layer_sizes = [input_dim] + layer_sizes + [latent_dim]
        
        layers = []
        for i in range(1, len(self.layer_sizes)):
            layers.append(torch.nn.Linear(in_features=self.layer_sizes[i-1], out_features=self.layer_sizes[i]))
            if i != len(self.layer_sizes) - 1:
                layers.append(torch.nn.LeakyReLU(.2))
        self.model = torch.nn.Sequential(*layers)
        
    def forward(self, X):
        return self.model(X)
    
    
class DenseDecoder(torch.nn.Module):
    
    def __init__(self, latent_dim, output_dim,
                 layer_sizes=[128, 512]):
        super(DenseDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.layer_sizes = [latent_dim] + layer_sizes + [output_dim]
        
        layers = []
        for i in range(1, len(self.layer_sizes)):
            layers.append(torch.nn.Linear(in_features=self.layer_sizes[i-1], out_features=self.layer_sizes[i]))
            if i != len(self.layer_sizes) - 1:
                layers.append(torch.nn.LeakyReLU(.2))
        self.model = torch.nn.Sequential(*layers)
        
    def forward(self, z):
        return self.model(z)
    

class ClusteringLayer(torch.nn.Module):
    
    def __init__(self, n_clusters, latent_dim, cluster_centers, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.cluster_centers = torch.nn.Parameter(cluster_centers)
        
    def forward(self, z):
        squared_norm = torch.sum((z.unsqueeze(1) - self.cluster_centers)**2, 2)
        numerator = (1.0 + squared_norm / self.alpha)**(-(self.alpha + 1) / 2)
        t_dist = (numerator.t() / torch.sum(numerator, 1)).t()
        return t_dist
    
    
class DEC(torch.nn.Module):
    
    def __init__(self, n_clusters, latent_dim, encoder, cluster_centers, alpha=1.0):
        super(DEC, self).__init__()
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.clustering_layer = ClusteringLayer(self.n_clusters, self. latent_dim, cluster_centers, alpha)
        
    def _target_distribution(q):
        weight = q**2 / torch.sum(q, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def encode(self, X):
        return self.encoder(X)
        
    def forward(self, X):
        z = self.encoder(X)
        return self.clustering_layer(z)


def get_label_mapping(y_true: NDArray, y_pred: NDArray) -> Dict[int, int]:
    cm = contingency_matrix(y_true, y_pred)
    true_labels, cluster_labels = linear_sum_assignment(cm, maximize=True)
    label_mapping = {cluster_l: true_l for cluster_l, true_l in zip(cluster_labels, true_labels)}
    return label_mapping


def train_and_select_model(X_train_full: NDArray, X_train_labeled: NDArray, y_train: NDArray,
                           n_models: int, dec_parameters: Optional[Dict[str, Any]]=None, verbose=True, device='cpu') -> DEC:
    if dec_parameters is None:
        dec_parameters = {}
    if verbose:
        print(f"Training {n_models} DEC models...")
        ts = time()
    models = [train_dec_end_to_end(X_train_full, verbose=False, device=device, **dec_parameters) for i in range(n_models)]
    if verbose:
        te = time()
        print(f"Training took {te - ts:.2f}s!\n")
        print(f"Selecting best performing model...")
        ts = time()

    max_acc = .0
    max_acc_i = -1
    for i, model in enumerate(models):
        y_pred_prob = model(torch.from_numpy(X_train_labeled).to(torch.float32).to(device)).detach().cpu().numpy()
        y_pred = np.argmax(y_pred_prob, axis=1)
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

    

def train_dec_end_to_end(X_train, X_val=None, dec_clustering_init='kmeans',
                         n_clusters=4, latent_dim=32, layer_sizes=[512, 128],
                         pretrain_args=None, dec_train_args=None,
                         verbose=True, device='cpu') -> DEC:
    input_dim = X_train.shape[-1]
    if pretrain_args is None:
        pretrain_args = {}
    if dec_train_args is None:
        dec_train_args = {}

    encoder = DenseEncoder(input_dim, latent_dim, layer_sizes=layer_sizes)
    decoder = DenseDecoder(latent_dim, input_dim, layer_sizes=layer_sizes[::-1])

    if verbose:
        print('Starting pretraining!')
    pretraining(encoder, decoder, X_train, X_val, verbose=verbose, device=device,
                **pretrain_args)
    if verbose:
        print('Pretraining finished.')
        print('Training DEC!')
    dec_model = train_dec(encoder, X_train, n_clusters, latent_dim, cluster_init=dec_clustering_init,
                          verbose=verbose, device=device, **dec_train_args)
    if verbose:
        print('DEC trained.')
    
    return dec_model


def pretraining(encoder, decoder,
                X_train, X_val=None, device='cpu',
                batch_size=128, n_epochs=50,
                optimizer='adam', lr=1E-3, weight_decay=2E-5,
                loss_fn='mse', noise_std=.0, verbose=True):
    if verbose:
        print(f"Pretraining using device: {device}")

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    training_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).to(torch.float32))
    training_dataloader = torch.utils.data.DataLoader(training_dataset,
                                                      batch_size=batch_size,
                                                      drop_last=True, shuffle=True)

    if X_val is not None:
        validation_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_val).to(torch.float32))
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset,
                                                            batch_size=8 * batch_size,
                                                            drop_last=True, shuffle=False)
    
    if optimizer == 'adam':
        optim = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr, weight_decay=2E-5)
    else:
        optim = torch.optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    
    if loss_fn == 'mse':
        loss_fn = torch.nn.functional.mse_loss
    else:
        loss_fn = torch.nn.functional.l1_loss
    
    for epoch in range(n_epochs):
        encoder.train()
        decoder.train()
        
        epoch_training_losses = []
        
        t_train_start = time()
        for X in iter(training_dataloader):
            X = X[0].to(device)
            
            optim.zero_grad()
            X_noise = X + torch.randn(X.size()).to(device) * noise_std
            z = encoder(X_noise)
            X_hat = decoder(z)
            loss = loss_fn(X, X_hat)
            loss.backward()
            optim.step()
            
            epoch_training_losses.append(loss.detach().item())
        t_train_end = time()

        if X_val is not None:
            encoder.eval()
            decoder.eval()

            epoch_validation_losses = []

            t_eval_start = time()
            with torch.no_grad():
                for X in iter(validation_dataloader):
                    X = X[0].to(device)
                    
                    z = encoder(X)
                    X_hat = decoder(z)
                    loss = loss_fn(X, X_hat)
                    
                    epoch_validation_losses.append(loss.detach().item())
            t_eval_end = time()
                    
        
        if verbose:
            print(f"Epoch {epoch + 1}/{n_epochs}\nTraining time: {t_train_end - t_train_start:.2f}s; Training loss: {np.mean(epoch_training_losses):.5f};")
            if X_val is not None:
                print(f"Validation time: {t_eval_end - t_eval_start:.2f}s; Evalutation loss: {np.mean(epoch_validation_losses):.5f};")
            print()


def train_dec(encoder, X_train,
              n_clusters=3, latent_dim=10,
              tol=1E-3, update_interval=300,
              batch_size=512, lr=1E-3, weight_decay=2E-5,
              device='cpu', cluster_init='kmeans', verbose=True):
    if verbose:
        print(f"Training using device: {device}")
    
    encoder = encoder.to(device)
    X_train = torch.from_numpy(X_train).to(torch.float32).to(device)
    
    # cluster init
    z_train = encoder(X_train).detach().cpu().numpy()
    if cluster_init == 'kmeans':
        if verbose:
            print('Initializing using KMeans!')
        kmeans = KMeans(n_clusters=n_clusters).fit(z_train)
        cluster_centers = torch.from_numpy(kmeans.cluster_centers_)
        current_cluster_assignment = kmeans.predict(z_train[np.random.choice(X_train.shape[0], size=batch_size, replace=False)])
    elif cluster_init == 'gmm':
        if verbose:
            print('Initializing using GMM!')
        gmm = GaussianMixture(n_components=n_clusters).fit(z_train)
        cluster_centers = torch.from_numpy(gmm.means_)
        current_cluster_assignment = gmm.predict(z_train[np.random.choice(X_train.shape[0], size=batch_size, replace=False)])
    prev_cluster_assignment = np.empty_like(current_cluster_assignment)
    delta = (current_cluster_assignment != prev_cluster_assignment).sum() / len(current_cluster_assignment)
    
    dec_model = DEC(n_clusters, latent_dim, encoder, cluster_centers).to(device)
    
    optimizer = torch.optim.Adam(dec_model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.KLDivLoss(reduction='batchmean')

    epoch = 0
    while delta > tol:
        dec_model.train()
        X_sub = X_train[np.random.choice(X_train.shape[0], size=batch_size, replace=False)]
        q = dec_model(X_sub)
        prev_cluster_assignment = current_cluster_assignment
        current_cluster_assignment = torch.argmax(q, dim=1).detach().cpu().numpy()

        if epoch % update_interval == 0:
            p = DEC._target_distribution(q).detach()

        loss = loss_fn(p.log(), q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % update_interval == 0:
                delta = (current_cluster_assignment != prev_cluster_assignment).sum() / len(current_cluster_assignment)
                if verbose:
                    print(f"Iter {epoch} - Loss: {loss.detach().cpu().item():.6f} - Delta: {delta:.6f}")
        epoch += 1

    return dec_model