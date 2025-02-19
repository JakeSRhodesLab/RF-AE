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
