import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()


def scale(X, x_min=0, x_max=1):
    nom = (X-X.min())*(x_max-x_min)
    denom = X.max() - X.min()
    denom = denom + (denom is 0)
    return x_min + nom/denom


df = pd.read_csv("Data/mnist_train.csv")
y = df.label
x = df.drop("label", axis=1)
x = x.values.reshape(-1, 28, 28)
x = pd.DataFrame(scale(x).reshape(-1, 784))
y = pd.DataFrame(enc.fit_transform(y.values.reshape(-1, 1)).toarray())

x.to_csv("Data/x_train.csv")
y.to_csv("Data/y_train.csv")

df_test = pd.read_csv("Data/mnist_test.csv")
y_test = df_test.label
x_test = df_test.drop("label", axis=1)
x_test = x_test.values.reshape(-1, 28, 28)
x_test = pd.DataFrame(scale(x_test).reshape(-1, 784))
y_test = pd.DataFrame(enc.fit_transform(
    y_test.values.reshape(-1, 1)).toarray())

x_test.to_csv("Data/x_test.csv")

y_test.to_csv("Data/y_test.csv")
