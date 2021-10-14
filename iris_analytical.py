from numpy.lib.npyio import load
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from numpy import array
from numpy.linalg import inv
from matplotlib import pyplot

data = load_iris()
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['species'] = data['target']

from sklearn.model_selection import train_test_split
x = df.drop('species', axis=1)
y = df.species
print(x.shape)
print(y.shape)
print(x)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
print('X_train: ' + str(x_train.shape))
print('Y_train: ' + str(y_train.shape))
print('X_test:  '  + str(x_test.shape))
print('Y_test:  '  + str(y_test.shape))

w = np.linalg.inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
print(w)
yhat = x_test.dot(w)

yhat = (yhat + 0.5).astype(np.int32)
y_test = (y_test + 0.5).astype(np.int32)
print(yhat.shape)
print(yhat.shape)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, yhat, normalize=False))