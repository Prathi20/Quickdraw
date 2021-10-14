from keras.datasets import mnist
import pandas as pd
import numpy as np

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X.reshape([-1, 784])
test_X = test_X.reshape([-1, 784])

w = np.linalg.inv(train_X.T.dot(train_X)).dot(train_X.T).dot(train_y)
print(w)
yhat = test_X.dot(w)
print(yhat.shape)
print(yhat.shape)

from sklearn.metrics import accuracy_score
print(accuracy_score(test_y, np.round(abs(yhat)), normalize=False))