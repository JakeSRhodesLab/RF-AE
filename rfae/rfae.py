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
from .torch_models import ProxAETorchModule, JSDivLoss
from rfae.utils.numpy_dataset import FromNumpyDataset
from rfae.utils.set_seeds import seed_everything



class RFAE():
    def __init__(self,
                 n_components=2,
                 batch_size=512,
                 lr=1e-3,
                 weight_decay=1e-5,
                 random_state=None,
                 device=None,
                 epochs=200,
                 hidden_dims=[800,400,100],
                 embedder_params=None,
                 lam=1e-2,
                 n_prototypes=500,
                 dropout_prob=0.0,
                 recon_loss_type='kl'):

        self.logger = logging.getLogger(__name__)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.n_components = n_components
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.epochs = epochs
        self.hidden_dims = hidden_dims
        self.lam = lam
        self.n_prototypes = n_prototypes
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

        self.embedder_params = embedder_params if embedder_params is not None else {'random_state': random_state,
                                                                                    'n_components': n_components,
                                                                                    'n_estimators': 500,
                                                                                    'prox_method': 'rfgap',
                                                                                    'oob_score': True,
                                                                                    'triangular': True,  # for original/oob prox methods only
                                                                                    'non_zero_diagonal': True,  # important for training stability
                                                                                    'normalize': False,  # whether to max normalize (0-1) the RF proximity matrix. False is recommended for extended RFGAP
                                                                                    'force_symmetric': False,  # important for better training/test consistency
                                                                                    'self_similarity': False,  # set to True for extremely noisy data, at the cost of destroying RFGAP properties
                                                                                    'batch_size': 'auto',  # batch size for RF proximity computation
                                                                                    'verbose': 0,
                                                                                    'n_jobs': -1,
                                                                                    }
        self.embedder_params['n_components'] = n_components  # ensure consistency with class param
        self.embedder_params['random_state'] = random_state  # ensure consistency with class param
        self.embedder = RFPHATE(**self.embedder_params)
        

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
        self.criterion_geo = nn.MSELoss()
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
        prox_sparse = self.embedder.proximity  # sparse (N x N) by default, unless specified as dense in embedder_params

        if self.n_prototypes is not None:
            classes = np.unique(y)
            k_per_class = max(1, self.n_prototypes // len(classes))
            prototype_indices = []
            
            for cls in classes:
                cls_indices = np.where(y == cls)[0]
                n_cls = len(cls_indices)
                if n_cls <= k_per_class:  # If small class, take all
                    prototype_indices.extend(cls_indices.tolist())
                    continue
                
                subset_size = min(n_cls, 30 * k_per_class)  # Dynamically determine subsample size based on desired number of prototypes
                rng = np.random.default_rng(self.random_state)
                chosen = rng.choice(cls_indices, size=subset_size, replace=False)
                sub_prox_sparse = prox_sparse[chosen][:, chosen]  # Extract sparse block and densify only this subset
                sub_prox = sub_prox_sparse.toarray().astype(np.float32)
                row_max = sub_prox.max(axis=1, keepdims=True)  # Normalize rows
                row_max[row_max == 0] = 1.0
                sub_prox /= row_max    
                sub_prox = 0.5 * (sub_prox + sub_prox.T)  # Symmetrize
                np.fill_diagonal(sub_prox, 1.0)
                sub_dist = 1.0 - sub_prox  # Convert similarities to distances
            
                # Run k-medoids
                km = kmedoids.KMedoids(k_per_class,method="fasterpam",random_state=self.random_state)
                km.fit(sub_dist)
            
                # Map medoids back to original indices
                medoids_global = chosen[km.medoid_indices_]
                prototype_indices.extend(medoids_global.tolist())
            self.prototype_indices = np.array(prototype_indices, dtype=int)
            self.logger.info(f"Selected {len(self.prototype_indices)} prototypes out of {self.input_shape} samples ({len(self.prototype_indices)/self.input_shape:.2%}).")
            self.training_proximities = prox_sparse[:, self.prototype_indices].toarray()  # Build training proximities: N × (#prototypes) (dense)

        else:
            # No prototype selection, use all samples. This can be memory-intensive for large datasets.
            self.prototype_indices = np.arange(self.input_shape)
            self.training_proximities = prox_sparse.toarray()   # Build training proximities: N × N (dense)

        self.init_torch_module(len(self.prototype_indices))
        self.optimizer = torch.optim.AdamW(self.torch_module.parameters(), lr=self.lr, weight_decay=self.weight_decay)

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
            if epoch % 20 == 0:
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
        