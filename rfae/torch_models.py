import torch.nn as nn
import torch.nn.functional as F

class LinearActivation(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob=0):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.elu(self.linear(x))
        x = self.dropout(x)
        return x

class LinearBlock(nn.Sequential):
    def __init__(self, dim_list, dropout_prob=0):
        modules = [LinearActivation(dim_list[i - 1], dim_list[i], dropout_prob) for i in range(1, len(dim_list) - 1)]
        modules.append(nn.Linear(dim_list[-2], dim_list[-1]))  # No activation for the last layer
        super().__init__(*modules)

class AETorchModule(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim, dropout_prob=0):
        super().__init__()

        full_list = [input_dim] + list(hidden_dims) + [z_dim]
        self.encoder = LinearBlock(dim_list=full_list, dropout_prob=dropout_prob)

        full_list.reverse()
        full_list[0] = z_dim
        self.decoder = LinearBlock(dim_list=full_list, dropout_prob=dropout_prob)

    def forward(self, x):
        z = self.encoder(x)
        z_decoder = z
        recon = self.decoder(z_decoder)
        return recon, z

class ProxAETorchModule(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim, output_activation='log_softmax', recon_dim=None, dropout_prob=0):
        """
        Args:
            input_dim (int): Dimension of input data
            hidden_dims (list of int): List of hidden layer dimensions
            z_dim (int): Latent space dimension
            output_activation (str): Type of output activation ('log_softmax', 'none', etc.)
            dropout_prob (float): Dropout probability
            recon_dim (int, optional): Dimension of reconstruction output. Defaults to input_dim.
        """
        super().__init__()

        if recon_dim is None:
            recon_dim = input_dim

        # Encoder
        full_list = [input_dim] + list(hidden_dims) + [z_dim]
        self.encoder = LinearBlock(dim_list=full_list, dropout_prob=dropout_prob)

        # Decoder
        decoder_list = [z_dim] + list(reversed(hidden_dims)) + [recon_dim]
        self.decoder = LinearBlock(dim_list=decoder_list, dropout_prob=dropout_prob)

        # Output activation
        self.output_activation = output_activation
        if output_activation == 'log_softmax':
            self.final_activation = nn.LogSoftmax(dim=1)
        elif output_activation == 'softmax':
            self.final_activation = nn.Softmax(dim=1)
        elif output_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif output_activation == 'none':
            self.final_activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported output_activation: {output_activation}")


    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        recon = self.final_activation(recon)
        return recon, z
    
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
