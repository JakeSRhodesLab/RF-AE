import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import graphtools
from scipy import sparse

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
                 hidden_dims=None,
                 embedder_params=None,
                 lam=1e-2,
                 dropout_prob=0.0,
                 recon_loss_type='jsd'):

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

        # Default parameters for RFPHATE
        default_embedder_params = {
            'random_state': random_state,
            'n_components': n_components,
            'n_landmark': 2000,  # performs well in general, cf PHATE paper
            'prox_method': 'rfgap',  # 'rfgap' or 'original'
            'model_type': 'rf',  # 'rf' (Random Forest) or 'et' (Extra Trees)
            'oob_score': False,  # set to True to monitor Random Forest performance
            'non_zero_diagonal': True,  # important for training stability
            'symm_mode': 'arithmetic',  # Direct built-in RF-GAP symmetrization (sparse block multiplication can introduce small asymmetries)
            'kernel_symm': None,  # disable PHATE internal symmetrization (which is heavier than RF-GAP's)
            'self_similarity': False,  # set to True for extremely noisy data, at the cost of destroying RFGAP properties
            'verbose': 0,
            'n_jobs': -1,
        }
        
        # Merge defaults with user overrides (if provided)
        if embedder_params is None:
            self.embedder_params = default_embedder_params
        else:
            self.embedder_params = {**default_embedder_params, **embedder_params}

        self.embedder = RFPHATE(**self.embedder_params)
        

    def init_torch_module(self):
        output_activation = {
            'kl': 'log_softmax',
            'jsd': 'softmax'
        }[self.recon_loss_type]

        self.logger.info(f"Initializing RF-AE module with output activation: {output_activation}")
        self.logger.info(f"Input shape: {self.input_shape}")

        self.hidden_dims_ratios = [0.4, 0.2, 0.05] # Default ratios
        if self.hidden_dims is None:
            # Dynamic calculation based on input size, determined by PHATE landmarks (fixed and relatively small)
            # Ensure they are integers and at least size of n_components + some buffer
            self.hidden_dims = [
                max(self.n_components * 2, int(self.input_shape * ratio)) 
                for ratio in self.hidden_dims_ratios
            ]
            self.logger.info(f"Dynamically set hidden_dims to: {self.hidden_dims}")

        self.torch_module = ProxAETorchModule(
            input_dim=self.input_shape,
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
        else:
            raise ValueError(f"Unknown recon_loss_type: {self.recon_loss_type}")

    
    def fit(self, x, y):
        self.labels = y

        if self.random_state is not None:
            seed_everything(self.random_state)

        self.z_target = self.embedder.fit_transform(x, y)

        if isinstance(self.embedder.phate_op.graph, graphtools.graphs.LandmarkGraph):
            transitions = self.embedder.phate_op.graph.transitions  # landmark graph transitions, shape (n_samples, n_landmarks)
        else:
            transitions = self.embedder.phate_op.graph.diff_op  # traditional graph transitions, shape (n_samples, n_samples)   

        self.input_shape = transitions.shape[1]
        self.init_torch_module()

        self.optimizer = torch.optim.AdamW(self.torch_module.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        transitions_tensor = torch.tensor(transitions.toarray(), dtype=torch.float) if sparse.issparse(transitions) else torch.tensor(transitions, dtype=torch.float)
        dataset = TensorDataset(transitions_tensor, torch.tensor(self.z_target, dtype=torch.float))
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.train_loop(self.torch_module, self.epochs, train_loader, self.optimizer, self.device)

        self.logger.info("Generating training embedding...")
        self.torch_module.eval()
        z_train = []
        
        # Use a sequential loader (shuffle=False) to maintain order
        eval_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for x_batch, _ in eval_loader:
                z_batch = self.torch_module.encoder(x_batch.to(self.device)).cpu().numpy()
                z_train.append(z_batch)
        
        self.embedding_ = np.concatenate(z_train)
        
        return self


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


    def transform(self, x):
        self.torch_module.eval()
        
        x = self.embedder.extend_to_data(x)  # shape (n_samples, n_landmarks) or (n_samples, n_samples)
        x = torch.tensor(x.toarray(), dtype=torch.float) if sparse.issparse(x) else torch.tensor(x, dtype=torch.float)
        
        loader = DataLoader(TensorDataset(x), batch_size=self.batch_size, shuffle=False)

        z = []
        with torch.no_grad():
            for batch in loader:
                z_batch = self.torch_module.encoder(batch[0].to(self.device)).cpu().numpy()
                z.append(z_batch)
        
        return np.concatenate(z)
    

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.embedding_


    def inverse_transform(self, x):
        self.torch_module.eval()
        x = FromNumpyDataset(x)
        loader = DataLoader(x, batch_size=self.batch_size, shuffle=False)
        x_hat = [self.torch_module.final_activation(self.torch_module.decoder(batch.to(self.device)))
                 .cpu().detach().numpy() for batch in loader]
        return np.concatenate(x_hat)
    

    def reconstruct(self, x):
        return self.inverse_transform(self.transform(x))
        