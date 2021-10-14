from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

data = load_iris()
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['species'] = data['target']

from sklearn.model_selection import train_test_split
x = df.drop('species', axis=1)
y = df.species
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

#linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(pred)
print(np.unique(y_test, return_counts = True))
print(np.unique(pred, return_counts = True))

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, np.round(abs(pred)), normalize=False))