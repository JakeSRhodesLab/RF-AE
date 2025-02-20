# RF-AE

Random Forest Autoencoders (RF-AE), a neural network-based framework for out-of-sample kernel extension that combines the flexibility of autoencoders with the supervised learning strengths of random forests and the geometry captured by RF-PHATE. RF-AE enables efficient out-of-sample supervised visualization. 

## Installation
Requirements: Python >= 3.9

```
pip install git+https://github.com/JakeSRhodesLab/RF-AE.git
```

## Quick Start
```
from rfae import RFAE

model = RFAE()
emb_train = model.fit_transform(x_train, y_train)   # training and get embeddings of training set
emb_test = model.transform(x_test)   # get embeddings of test set
```

## Parameters

```n_components```: Dimensionality of the embedding space.

```lam```: Main hyper-parameter that controls the trade-off between RF-GAP neighborhood reconstruction and geometric loss on the bottleneck. Valid values are between ```0``` and ```1```: ```lam=0``` trains a deep encoder-only network to regress onto precomputed training RF-PHATE embeddings while ```lam=1``` trains an autoencoder reconstructing RF-GAP neighborhoods without geometric regularization. Default: ```1e-3```.

```loss_scaling```: If ```True```, scales the embedding loss and reconstruction loss to be on the same scale. In this case, lam should be between ```0``` and ```1```. Default: ```False```.

```pct_landmark```: Specifies the number or percentage of landmarks used for proximity calculations during training. ```"auto"```: Uses 2000 landmarks. A float between 0 and 1: Represents the percentage of the dataset size to use as landmarks. ```None```: Uses all proximities. Default: ```"auto"```.

```n_pca```: Number of principal components (PCs) used for selecting landmarks. Must be smaller than the dataset size. ```None``` is default (100) from PHATE

```embedder_params```: parameters dict for [RF-PHATE](https://github.com/jakerhodes/RF-PHATE.git)

```hidden_dims```: List of dimensions for the hidden layers of the autoencoder.

```dropout_prob```: Dropout probability applied during training to prevent overfitting.
 
```lr```: Learning rate for model optimization.

```weight_decay```: Regularization term for weight updates to prevent overfitting.

```batch_size```: Number of samples per batch during training.

```epochs```: Total number of training epochs.

```random_state```: Seed for random number generation to ensure reproducibility.

```device```: Computing device used for training. 

```early_stopping```: If ```True```, enables early stopping during training to prevent overfitting. Default: ```False```.

```patience```: Number of epochs to wait without improvement before stopping training. Only used when ```early_stopping = True```.

```delta_factor```: Minimum relative improvement required to reset the early stopping counter. Only used when ```early_stopping = True```.

```save_model```: If True, saves the model after training. Only used when ```early_stopping = True```.






