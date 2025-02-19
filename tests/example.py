'''
A simple test to check if the model is loading fine.
'''
from rfae import RFAE 
import numpy as np

x = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)
x_train, x_test = x[:800], x[800:]
y_train, y_test = y[:800], y[800:]

model = RFAE()
emb_train = model.fit_transform(x_train, y_train)
emb_test = model.transform(x_test)

print(emb_train.shape, emb_test.shape)
