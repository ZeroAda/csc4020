from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("Carseats.csv")

d_x = data.iloc[:,1:]
d_y = data.iloc[:,0]

x_train = d_x[:300,:]
y_train = d_y[:300,:]

x_test = d_x[300:,:]
y_test = d_y[300:,:]

# Decision Tree
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
dt.score(x_test,y_test)

# Bagging

# Random Forest

# AdaBoost
