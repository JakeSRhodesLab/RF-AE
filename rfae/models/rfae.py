import os
from ..utils.numpy_dataset import FromNumpyDataset
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
from .rfphate import RFPHATE
from .base_model import BaseModel
from .torch_models import ProxAETorchModule, EarlyStopping
import numpy as np
from scipy import sparse
import graphtools
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from ..utils.set_seeds import seed_everything
import logging
from typing import Union, Optional, Dict, Any


class RFAE(BaseModel):

    """
    This model takes the row normalized RF proximities as input and attempts to reconstruct them.
    Adds RF-PHATE Geometric Regularization into the bottleneck layer with hyperparameter lam between
    0 (no reconstruction) and 1 (reconstruction only)
    """

    def __init__(self,
                 n_components: int = 2,
                 lr: float = 1e-3,
                 batch_size: int = 64,
                 weight_decay: float = 1e-5,
                 random_state: int = 42,
                 device: str = 'cpu',
                 epochs: int = 200,
                 n_pca: Optional[int] = None,
                 hidden_dims: list = [800,400,100],
                 embedder_params: Optional[Dict[str, Any]] = None,
                 lam: float = 1e-2,
                 loss_scaling: bool = False,
                 pct_landmark: Union[str, float, None] = 'auto',   # "auto" or float between 0 and 1 or None
                 dropout_prob: float = 0.0,
                 early_stopping: bool = False,   # only for early stopping
                 patience: int = 50,   # only use for early_stopping = True
                 delta_factor: float = 1e-3,   # only use for early_stopping = True
                 save_model: bool = False   # only use for early_stopping = True
                 ):

        self.n_components = n_components
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device
        self.epochs = epochs
        self.n_pca = n_pca
        self.hidden_dims = hidden_dims
        self.lam = lam
        self.loss_scaling = loss_scaling
        self.pct_landmark = pct_landmark
        self.dropout_prob = dropout_prob
        self.early_stopping = early_stopping
        self.patience = patience   
        self.delta_factor = delta_factor
        self.save_model = save_model

        if embedder_params is None:
            self.embedder = RFPHATE(random_state = random_state, n_components = self.n_components)
        else: 
            self.embedder = RFPHATE(random_state = random_state, n_components = self.n_components, **embedder_params)

    def init_torch_module(self, input_shape):

        self.torch_module = ProxAETorchModule(input_dim   = input_shape,
                                              hidden_dims = self.hidden_dims,
                                              z_dim       = self.n_components,
                                              dropout_prob = self.dropout_prob)


    def fit(self, x, y):
        
        self.input_shape = x.shape[0]
        self.labels = y

        self.z_target = self.embedder.fit_transform(x, y)
        self.training_proximities = self.embedder.proximity

        if self.random_state is not None:
            seed_everything(self.random_state)
        
        if self.pct_landmark is not None:
            if self.pct_landmark == 'auto':
                # Use clusters computed by Landmark PHATE (PCA on diffusion operator, then clustering )
                if isinstance(self.embedder .phate_op.graph, graphtools.graphs.LandmarkGraph):
                    self.cluster_labels = self.embedder.phate_op.graph.clusters
                else:
                    self.cluster_labels = range(self.input_shape)  # No clusters
                n_landmarks = len(np.unique(self.cluster_labels))
            elif 0 < self.pct_landmark < 1:
                n_landmarks = int(self.pct_landmark*self.input_shape)
                if self.n_pca is None:
                    self.n_pca = self.embedder.phate_op.n_pca
                if self.n_pca >= self.input_shape:
                    print('The number of PCs \'n_pca\' to retain exceeds the training size.')
                    self.n_pca = self.input_shape - 1
                # Perform k-means clustering after PCA on the diffusion operator (just like in PHATE)
                pca = TruncatedSVD(n_components=self.n_pca, random_state=self.random_state)
                row_sums = np.array(self.training_proximities.sum(axis=1)).flatten()
                row_sums[row_sums == 0] = 1.0    # Avoid division by zero (set rows with sum 0 to 1 to keep them as zero rows after normalization)
                inv_row_sums = sparse.diags(1.0 / row_sums)
                normalized_training_proximities = inv_row_sums @ self.training_proximities
                pca_transformed = pca.fit_transform(normalized_training_proximities)
                clustering = KMeans(n_clusters=n_landmarks, 
                                    random_state=self.random_state).fit(pca_transformed)
                self.cluster_labels = clustering.labels_
            else:
                raise ValueError("pct_landmark of RF-AE must be between 0 and 1 or 'auto' or None")
            # Update the training proximities so that each column represent the aggegated proximities between each
            # instance (row) and cluster (column), so that final shape = (n_samples, n_landmarks)
            self.training_proximities = sparse.hstack(
                [
                    sparse.csr_matrix(self.training_proximities[:, self.cluster_labels == i].sum(axis=1))
                    for i in np.unique(self.cluster_labels)
                ]
            ).toarray()  # Convert back to dense

            self.init_torch_module(n_landmarks)
        
        else:
            self.cluster_labels = None
            self.training_proximities = self.training_proximities.toarray()
            self.init_torch_module(self.input_shape)


        self.optimizer = torch.optim.AdamW(self.torch_module.parameters(),
                                              lr=self.lr,
                                              weight_decay=self.weight_decay)

        self.criterion_recon = nn.KLDivLoss(reduction="batchmean")
        self.criterion_geo = nn.MSELoss()

        # Row-normalized Tensor proximities
        training_proximities = torch.tensor(self.training_proximities, dtype=torch.float)
        training_proximities = F.normalize(training_proximities, p=1)

        # Training dataset
        tensor_dataset = TensorDataset(training_proximities, torch.tensor(self.z_target, dtype=torch.float))

        train_loader = DataLoader(tensor_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.train_loop(self.torch_module, self.epochs, train_loader, self.optimizer, self.device)

    def fit_transform(self, x, y):
        self.fit(x, y)
        self.z_latent = self.transform(self.training_proximities, precomputed=True)
        return self.z_latent


    def compute_loss(self, x, x_hat, z_target, z):

        loss_recon = self.criterion_recon(x_hat, x)
        loss_emb = self.criterion_geo(z_target, z)

        self.recon_loss_temp = loss_recon.item()
        self.emb_loss_temp = loss_emb.item()

        # Dynamic scaling factors for balancing magnitudes
        # Compute scaling factors only once and store them
        if self.loss_scaling and (not hasattr(self, "scale_recon") or not hasattr(self, "scale_emb")):
            self.scale_recon = 1 / (loss_recon.detach().mean() + 1e-8)
            self.scale_emb = 1 / (loss_emb.detach().mean() + 1e-8)
        else:
            self.scale_recon = 1
            self.scale_emb = 1
        balanced_loss = self.lam * loss_recon * self.scale_recon + (1 - self.lam) * loss_emb * self.scale_emb
        self.balanced_loss = balanced_loss.item()
        return balanced_loss

    def train_loop(self, model, epochs, train_loader, optimizer, device = 'cpu'):
        self.epoch_losses_recon = []
        self.epoch_losses_emb = []
        self.epoch_losses_balanced = []
        best_loss = float("inf")
        counter=0

        model.to(device)
        model.train()

        for epoch in range(epochs):
            running_recon_loss = 0
            running_emb_loss = 0
            running_balanced_loss = 0

            for x, z_target in train_loader:
                x = x.to(device)
                z_target = z_target.to(device)

                recon, z = model(x)

                optimizer.zero_grad()
                self.compute_loss(x, recon, z, z_target).backward()

                running_recon_loss += self.recon_loss_temp
                running_emb_loss += self.emb_loss_temp
                running_balanced_loss += self.balanced_loss

                optimizer.step()

            # Track losses per epoch
            self.epoch_losses_recon.append(running_recon_loss / len(train_loader))
            self.epoch_losses_emb.append(running_emb_loss / len(train_loader))
            self.epoch_losses_balanced.append(running_balanced_loss / len(train_loader))

            # Periodic logging of losses
            if epoch % 50 == 0:
                logging.info(f"Epoch {epoch}/{self.epochs}, Recon Loss: {self.epoch_losses_recon[-1]:.7f}, Geo Loss: {self.epoch_losses_emb[-1]:.7f}")

            # Check for early stopping
            if self.early_stopping:
                os.makedirs(f"{self.random_state}/", exist_ok=True)
                subfolder_path = f"{self.random_state}/best_{self.random_state}.pth"
                early_stopping = EarlyStopping(patience = self.patience,
                                        delta_factor = self.delta_factor, 
                                        save_model = self.save_model, 
                                        save_path = subfolder_path)
                should_stop, best_loss, counter = early_stopping(self.epoch_losses_balanced[-1], best_loss, counter, model)
                if should_stop:
                    logging.info(f"Stopping training early at epoch {epoch}")
                    return  

    def evaluate_recon(self, x, precomputed=False):
        self.torch_module.eval()
        total_kl_div  = 0
        total_samples = 0

        if not precomputed:
            x = self.embedder.prox_extend(x).toarray()
        
        x = torch.tensor(x, dtype=torch.float)
        x = F.normalize(x, p=1)
        
        loader = DataLoader(TensorDataset(x), batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for x_batch in loader:
                batch_size = x_batch.size(0)
                x_batch.to(self.device)

                recon, _ = self.torch_module(x)

                total_kl_div  += self.criterion_recon(recon, x).item() * batch_size
                total_samples += batch_size
        
        return total_kl_div / total_samples


    def transform(self, x, precomputed=False):
        self.torch_module.eval()

        if not precomputed:
            x = self.embedder.prox_extend(x)

            if self.cluster_labels is not None:
                # Aggregate distance between training clusters to fit training input dimension
                x = sparse.hstack(
                [
                    sparse.csr_matrix(x[:, self.cluster_labels == i].sum(axis=1))
                    for i in np.unique(self.cluster_labels)
                ]
                ).toarray()  # Convert back to dense
            else:
                x = x.toarray()

        x = torch.tensor(x, dtype=torch.float)
        x = F.normalize(x, p=1)
        
        loader = DataLoader(TensorDataset(x), batch_size=self.batch_size, shuffle=False)

        z = []
        with torch.no_grad():
            for batch in loader:
                z_batch = self.torch_module.encoder(batch[0].to(self.device)).cpu().numpy()
                z.append(z_batch)
        
        return np.concatenate(z)


    def inverse_transform(self, x):
        self.torch_module.eval()
        x = FromNumpyDataset(x)
        loader = DataLoader(x, batch_size=self.batch_size, shuffle=False)
        x_hat = [self.torch_module.log_softmax(self.torch_module.decoder(batch.to(self.device)))
                 .cpu().detach().numpy() for batch in loader]

        return np.concatenate(x_hat)
        