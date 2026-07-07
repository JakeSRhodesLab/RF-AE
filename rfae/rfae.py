import sys
import torch
import torch.nn as nn
import numpy as np
import logging
import graphtools
from numbers import Integral
from scipy import sparse

from torch.utils.data import TensorDataset, DataLoader
from rfphate import RFPHATE
from .torch_models import ProxAETorchModule, JSDivLoss, PotentialLoss
from rfae.utils.numpy_dataset import FromNumpyDataset
from rfae.utils.set_seeds import seed_everything


class RFAE():
    def __init__(self,
                 n_components=2,
                 t='auto',
                 batch_size=512,
                 lr=1e-3,
                 weight_decay=1e-5,
                 random_state=None,
                 device=None,
                 epochs=200,
                 hidden_dims=None,
                 embedder_params=None,
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
        self.t = t
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

        default_embedder_params = {
            'random_state': random_state,
            'n_jobs': -1,
            'proximity_params': {
                'weight_scheme': 'gap',
            },
            'phate_params': {
                'n_components': n_components,
                't': t,
                'n_landmark': 2000,  # performs well in general, cf PHATE paper
                'kernel_symm': None,  # disable PHATE internal symmetrization by default
                'verbose': 1,
            },
        }

        self.embedder_params = default_embedder_params if embedder_params is None else dict(embedder_params)
        self.embedder_params.setdefault('phate_params', {})
        self.embedder_params['phate_params'] = dict(self.embedder_params['phate_params'])
        self.embedder_params['phate_params']['t'] = t
        self.embedder = RFPHATE(**self.embedder_params)


    def init_torch_module(self):
        output_activations = {
            'kl': 'log_softmax',
            'jsd': 'softmax',  # JSD is bounded, so easier to optimize in a coupled prox-feature space model
            'potential': 'softmax',  # better for a direct connection to PHATE
        }
        if self.recon_loss_type not in output_activations:
            raise ValueError(f"Unknown recon_loss_type: {self.recon_loss_type}")
        output_activation = output_activations[self.recon_loss_type]

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
        if self.recon_loss_type == 'kl':
            self.criterion_recon = nn.KLDivLoss(reduction="batchmean")
        elif self.recon_loss_type == 'jsd':
            self.criterion_recon = JSDivLoss(reduction='batchmean')
        elif self.recon_loss_type == 'potential':
            self.criterion_recon = PotentialLoss(reduction='mean')
        else:
            raise ValueError(f"Unknown recon_loss_type: {self.recon_loss_type}")
    
    
    def _build_transition_matrices(self, fitted_phate_op):
        graph = fitted_phate_op.graph
        t = fitted_phate_op._find_optimal_t() if self.t == "auto" else self.t
    
        if isinstance(graph, graphtools.graphs.LandmarkGraph):
            point_to_landmark = self._to_numpy(graph.transitions)
            landmark_to_landmark = self._to_numpy(graph.landmark_op)
            target = point_to_landmark @ self._aggregate_transition_powers(
                landmark_to_landmark, t
            )
            return point_to_landmark, self._row_normalize(target)
    
        point_to_point = self._to_numpy(graph.diff_op)
        target = self._aggregate_transition_powers(point_to_point, t)
    
        return point_to_point, self._row_normalize(target)
    
    
    @staticmethod
    def _aggregate_transition_powers(matrix, t):
        matrix = np.asarray(matrix)
    
        current = matrix.copy()
        aggregate = current.copy()
    
        for _ in range(2, t + 1):
            current = current @ matrix
            aggregate += current
    
        return aggregate
    
    
    @staticmethod
    def _row_normalize(matrix):
        matrix = np.asarray(matrix)
        row_sums = matrix.sum(axis=1, keepdims=True)
    
        return matrix / np.where(row_sums == 0, 1.0, row_sums)


    @staticmethod
    def _to_numpy(matrix):
        return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)

    
    def fit(self, x, y, adjust_diagonal=True, force_symmetric=True):
        self.labels = y

        if self.random_state is not None:
            seed_everything(self.random_state)

        self.embedder.fit(
            x,
            y,
            adjust_diagonal=adjust_diagonal,
            force_symmetric=force_symmetric,
        )

        phate_op = self.embedder.phate_op_
        transitions, transition_target = self._build_transition_matrices(phate_op)

        self.input_shape = transitions.shape[1]
        self.init_torch_module()

        self.optimizer = torch.optim.AdamW(self.torch_module.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        transitions_tensor = torch.tensor(transitions.toarray(), dtype=torch.float) if sparse.issparse(transitions) else torch.tensor(transitions, dtype=torch.float)
        transition_target_tensor = torch.tensor(transition_target.toarray(), dtype=torch.float) if sparse.issparse(transition_target) else torch.tensor(transition_target, dtype=torch.float)
        dataset = TensorDataset(transitions_tensor, transition_target_tensor)
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


    def compute_loss(self, x_hat, x_target):
        loss_recon = self.criterion_recon(x_hat, x_target)

        self.recon_loss_temp = loss_recon.item()
        return loss_recon


    def train_loop(self, model, epochs, train_loader, optimizer, device = 'cpu'):
        self.epoch_losses_recon = []

        model.to(device)
        model.train()

        for epoch in range(epochs):
            running_recon_loss = 0

            for x, x_target in train_loader:
                x = x.to(device)
                x_target = x_target.to(device)

                recon, _ = model(x)

                optimizer.zero_grad()
                self.compute_loss(recon, x_target).backward()

                running_recon_loss += self.recon_loss_temp

                optimizer.step()

            # Track losses per epoch
            self.epoch_losses_recon.append(running_recon_loss / len(train_loader))

            # Periodic logging of losses
            if epoch % 50 == 0:
                self.logger.info(
                    f"Epoch {epoch}/{epochs} "
                    f"- Recon Loss: {self.epoch_losses_recon[-1]:.7f}"
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
    

    def fit_transform(self, x, y, adjust_diagonal=True, force_symmetric=True):
        self.fit(
            x,
            y,
            adjust_diagonal=adjust_diagonal,
            force_symmetric=force_symmetric,
        )
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
