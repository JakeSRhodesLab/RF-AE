import torch
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

class ProxAETorchModule(AETorchModule):
    def __init__(self, input_dim, hidden_dims, z_dim, output_activation='log_softmax', dropout_prob=0):
        super().__init__(input_dim, hidden_dims, z_dim, dropout_prob)
        self.log_softmax = nn.LogSoftmax(dim=1)  # Log Softmax Output Layer for numerical stability with nn.KLDivLoss
        self.output_activation = output_activation

    def forward(self, x):
        z = self.encoder(x)
        z_decoder = z
        recon = self.decoder(z_decoder)
        recon = self.log_softmax(recon)  # Apply log softmax activation to the output
        return recon, z

class MLPReg(nn.Module):
    def __init__(self, encoder, input_dim, output_dim):
        super().__init__()

        self.encoder = encoder
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        z = self.encoder(x)
        pred = self.linear(z)
        return pred
    
class EarlyStopping:
    def __init__(self, patience=5, delta_factor=0.01, save_model=False, save_path="best_model.pth"):
        self.patience = patience
        self.delta_factor = delta_factor   # Set delta as a percentage of loss
        self.save_model = save_model
        self.save_path = save_path

    def __call__(self, current_loss, best_loss,counter, model):
        # Save the best model if the current loss is the lowest so far
        if best_loss is None:
            best_loss = current_loss   # first epoch

        dynamic_delta = best_loss * self.delta_factor

        if best_loss - current_loss > dynamic_delta:
            counter = 0  # Reset patience counter because there's improvement
        else:
            counter += 1  # No significant improvement, increase counter

        # save model if the current loss is the lowest so far
        if best_loss > current_loss:
            best_loss = current_loss
            if self.save_model: 
                torch.save(model, self.save_path)
        
        return counter >= self.patience, best_loss, counter  # Stop training if patience is exceeded