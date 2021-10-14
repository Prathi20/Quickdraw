from keras.datasets import mnist
import pandas as pd
import numpy as np

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X.reshape([-1, 784])
test_X = test_X.reshape([-1, 784])
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

"""
from matplotlib import pyplot
from matplotlib import pyplot
for i in range(9):  
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
    pyplot.show()
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(train_X, train_y)
pred = model.predict(test_X)
print(pred)

test_y = test_y.astype(np.float32)
pred = pred.astype(np.float32)

print(np.unique(test_y, return_counts = True))
print(np.unique(pred, return_counts = True))

from sklearn.metrics import accuracy_score

print(accuracy_score(test_y, np.round(abs(pred)), normalize=False))