import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

data = pd.read_csv("california_housing.csv")

x = data[['AveRooms']].values
y = data['MedHouseVal'].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

mean = x_train.mean()
std = x_train.std()

x_train = (x_train - mean)/std
x_test = (x_test - mean)/std

x_train_b = np.c_[np.ones(len(x_train)),x_train]
x_test_b = np.c_[np.ones(len(x_test)),x_test]

theta_gd = np.zeros(2)
alpha = 0.01
iterations = 10000
m = len(y_train)

for i in range(iterations):
    y_pred = x_train_b @ theta_gd
    gradient = (2/m) * (x_train_b.T @(y_pred - y_train))
    theta_gd = theta_gd - alpha * gradient

y_pred_gd = x_test_b @ theta_gd

print("Gradient Descent")
print("Mean squared error: ",mean_squared_error(y_test,y_pred_gd))
print("R squared: ",r2_score(y_test,y_pred_gd))

theta_ne = np.linalg.pinv(x_train_b.T @ x_train_b) @ x_train_b.T @ y_train

y_pred_ne = x_test_b @ theta_ne

print("Normal Equation")
print("Mean Squared Error: ",mean_squared_error(y_test,y_pred_ne))
print("R Squared: ",r2_score(y_test,y_pred_ne))

idx = np.argsort(x_test[:,0])

plt.scatter(x_test,y_test,label = "Actual Data")

plt.plot(x_test[idx],y_pred_gd[idx],color = "red",label = "Gradient Descent")
plt.plot(x_test[idx],y_pred_ne[idx],linestyle = "--",color = "green",label="Normal Equation")

plt.xlabel("Average Rooms")
plt.ylabel("Median House Value")
plt.title("Linear Regression - California Housing")
plt.legend()
plt.show()