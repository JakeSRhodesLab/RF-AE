import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import kmedoids

from torch.utils.data import TensorDataset, DataLoader
from rfphate import RFPHATE
from .torch_models import ProxAETorchModule
from rfae.utils.numpy_dataset import FromNumpyDataset
from rfae.utils.set_seeds import seed_everything


class JSDivLoss(nn.Module):
    def __init__(self, reduction='batchmean', eps=1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, p, q):
        p = p.clamp(min=self.eps, max=1.0)
        q = q.clamp(min=self.eps, max=1.0)
        m = 0.5 * (p + q)
        return 0.5 * (
            F.kl_div(m.log(), p, reduction=self.reduction) +
            F.kl_div(m.log(), q, reduction=self.reduction)
        )


class RFAE():
    def __init__(self,
                 n_components=2,
                 lr=1e-3,
                 batch_size=512,
                 weight_decay=1e-5,
                 random_state=None,
                 device=None,
                 epochs=200,
                 hidden_dims=[800,400,100],
                 embedder_params=None,
                 lam=1e-2,
                 pct_prototypes=0.02,
                 dropout_prob=0.0,
                 recon_loss_type='jsd'):
        
        # ---------------------------
        # Logger initialization
        # ---------------------------
        self.logger = logging.getLogger(__name__)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # ---------------------------
        # Model configuration
        # ---------------------------

        self.n_components = n_components
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.epochs = epochs
        self.hidden_dims = hidden_dims
        self.lam = lam
        self.pct_prototypes = pct_prototypes
        self.dropout_prob = dropout_prob
        self.recon_loss_type = recon_loss_type.lower()

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(): # For Apple Silicon
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.logger.info(f"Using device: {self.device}")

        self.embedder_params = embedder_params if embedder_params is not None else {
                                                                                    'n_estimators': 500,
                                                                                    'prox_method': 'rfgap',
                                                                                    'oob_score': True,
                                                                                    'normalize': False,
                                                                                    'force_symmetric': False,  # important for better training/test consistency
                                                                                    'verbose': 0,
                                                                                    'n_jobs': -1
                                                                                    }
        self.embedder = RFPHATE(random_state=random_state, n_components=n_components, **self.embedder_params)
        

    def init_torch_module(self, input_shape):
        output_activation = {
            'kl': 'log_softmax',
            'jsd': 'softmax',
            'mse': 'softmax'
        }[self.recon_loss_type]

        self.torch_module = ProxAETorchModule(
            input_dim=input_shape,
            hidden_dims=self.hidden_dims,
            z_dim=self.n_components,
            dropout_prob=self.dropout_prob,
            output_activation=output_activation
        )

        if self.recon_loss_type == 'kl':
            self.criterion_recon = nn.KLDivLoss(reduction="batchmean")
        elif self.recon_loss_type == 'jsd':
            self.criterion_recon = JSDivLoss(reduction='batchmean')
        elif self.recon_loss_type == 'mse':
            self.criterion_recon = nn.MSELoss(reduction="mean")
        else:
            raise ValueError(f"Unknown recon_loss_type: {self.recon_loss_type}")

    def fit(self, x, y):
        self.input_shape = x.shape[0]
        self.labels = y

        if self.random_state is not None:
            seed_everything(self.random_state)

        self.z_target = self.embedder.fit_transform(x, y)
        self.training_proximities = self.embedder.proximity.toarray()
        
        if 0 < self.pct_prototypes < 1:
            # Max normalize each row, symmetrize, set diagonal to 1
            row_max = self.training_proximities.max(axis=1, keepdims=True)
            row_max[row_max == 0] = 1.0
            prox_matrix = self.training_proximities / row_max
            prox_matrix = 0.5 * (prox_matrix + prox_matrix.T)
            np.fill_diagonal(prox_matrix, 1.0)
            
            # Create distance matrix
            dist_matrix = 1 - prox_matrix

            k = int(self.pct_prototypes * self.input_shape)
            classes = np.unique(y)
            k_per_class = max(1, k // len(classes))

            prototype_indices = []
            for cls in classes:
                cls_indices = np.where(y == cls)[0]
                if len(cls_indices) <= k_per_class:
                    prototype_indices.extend(cls_indices)
                    continue

                sub_dists = dist_matrix[np.ix_(cls_indices, cls_indices)]
                km = kmedoids.KMedoids(k_per_class, method='fasterpam', random_state=self.random_state)
                km.fit(sub_dists)
                prototype_indices.extend(cls_indices[km.medoid_indices_])

            self.prototype_indices = np.array(prototype_indices, dtype=int)
            self.training_proximities = self.training_proximities[:, self.prototype_indices]
            self.init_torch_module(len(self.prototype_indices))
        else:
            self.prototype_indices = np.arange(self.input_shape)
            self.init_torch_module(self.input_shape)

        self.optimizer = torch.optim.AdamW(self.torch_module.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion_geo = nn.MSELoss()

        training_proximities = torch.tensor(self.training_proximities, dtype=torch.float)
        training_proximities = F.normalize(training_proximities, p=1)

        train_loader = DataLoader(TensorDataset(training_proximities, torch.tensor(self.z_target, dtype=torch.float)),
                                  batch_size=self.batch_size, shuffle=True)

        self.train_loop(self.torch_module, self.epochs, train_loader, self.optimizer, self.device)


    def compute_loss(self, x, x_hat, z, z_target):
        loss_recon = self.criterion_recon(x_hat, x)
        loss_emb = self.criterion_geo(z_target, z)

        self.recon_loss_temp = loss_recon.item()
        self.emb_loss_temp = loss_emb.item()

        balanced_loss = self.lam * loss_recon + (1 - self.lam) * loss_emb
        self.balanced_loss = balanced_loss.item()
        return balanced_loss

    def train_loop(self, model, epochs, train_loader, optimizer, device = 'cpu'):
        self.epoch_losses_recon = []
        self.epoch_losses_emb = []
        self.epoch_losses_balanced = []

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
                self.logger.info(
                    f"Epoch {epoch}/{epochs} "
                    f"- Recon Loss: {self.epoch_losses_recon[-1]:.7f} "
                    f"- Geo Loss: {self.epoch_losses_emb[-1]:.7f}"
                )


    def transform(self, x, precomputed=False):
        self.torch_module.eval()

        if not precomputed:
            x = self.embedder.prox_extend(x, self.prototype_indices).toarray()
 

        x = torch.tensor(x, dtype=torch.float)
        x = F.normalize(x, p=1)
        
        loader = DataLoader(TensorDataset(x), batch_size=self.batch_size, shuffle=False)

        z = []
        with torch.no_grad():
            for batch in loader:
                z_batch = self.torch_module.encoder(batch[0].to(self.device)).cpu().numpy()
                z.append(z_batch)
        
        return np.concatenate(z)
    

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(self.training_proximities, precomputed=True)


    def inverse_transform(self, x):
        self.torch_module.eval()
        x = FromNumpyDataset(x)
        loader = DataLoader(x, batch_size=self.batch_size, shuffle=False)
        x_hat = [self.torch_module.final_activation(self.torch_module.decoder(batch.to(self.device)))
                 .cpu().detach().numpy() for batch in loader]
        return np.concatenate(x_hat)
    

    def reconstruct(self, x):
        return self.inverse_transform(self.transform(x))
        