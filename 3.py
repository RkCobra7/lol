import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('diabetes.csv')

x = data.drop('target',axis = 1)
y = data['target']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

linear = LinearRegression()
linear.fit(x_train,y_train)
linear_pred = linear.predict(x_test)

print("Linear Model")
print("Mean Squared Error: ",round(mean_squared_error(y_test,linear_pred),4))
print("R Squared ",round(r2_score(y_test,linear_pred),4))

alphas = [0.001,0.01,0.1,1,10,100]

ridge_grid = GridSearchCV(Ridge(),{'alpha':alphas},cv = 5,scoring = 'neg_mean_squared_error')
ridge_grid.fit(x_train,y_train)
ridge_pred = ridge_grid.predict(x_test)

print("Ridge Model")
print("Mean Squared Error: ",round(mean_squared_error(y_test,ridge_pred),4))
print("R Squared ",round(r2_score(y_test,ridge_pred),4))

lasso_grid = GridSearchCV(Lasso(max_iter = 10000),{'alpha':alphas},cv = 5,scoring = 'neg_mean_squared_error')
lasso_grid.fit(x_train,y_train)
lasso_pred = lasso_grid.predict(x_test)

print("Lasso Model")
print("Mean Squared Error: ",round(mean_squared_error(y_test,lasso_pred),4))
print("R Squared ",round(r2_score(y_test,lasso_pred),4))

names = ['Linear','Ridge','Lasso']
preds = [linear_pred,ridge_pred,lasso_pred]
mse_scores = [mean_squared_error(y_test,p) for p in preds]
r2_scores = [r2_score(y_test,p) for p in preds]

print("Best Ridge alpha: ",ridge_grid.best_params_['alpha'])
print("Best Lasso alpha: ",lasso_grid.best_params_['alpha'])
plt.figure(figsize = (10,6))

for i in range(3):
    plt.scatter(y_test,preds[i],label = names[i])
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(figsize = (10,6))
plt.plot(names,mse_scores,label = "MSE")
plt.xlabel("Models")
plt.ylabel("MSE Score")
plt.title("MSE performance")

plt.figure(figsize = (10,6))
plt.plot(names,r2_scores,label = "R2")
plt.xlabel("Models")
plt.ylabel("R2 Score")
plt.title("R2 performance")

plt.show()
