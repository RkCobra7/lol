import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

data = pd.read_csv("winequality-red.csv")
x = data.drop('quality',axis = 1)
y = data['quality']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

architectures = {
    '1 Layer (32)' : (32,),
    '1 Layer (64)' : (64,),
    '2 Layer (64,32)' : (64,32),
    '2 Layer (128,64)' : (128,64),
    '3 Layer (128,64,32)' : (128,64,32),
}

rmse_scores = []
mae_scores = []
r2_scores = []

for name, layer in architectures.items():
    model = MLPRegressor(hidden_layer_sizes = layer,activation = 'relu',solver = 'adam',random_state = 42)
    start = time.time()
    model.fit(x_train,y_train)
    end = time.time()
    y_pred = model.predict(x_test)
    tr_time = round(end - start,2)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    mae = mean_absolute_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    r2_scores.append(r2)
    print(name)
    print("RMSE: ",rmse)
    print("MAE: ",mae)
    print("R2 Score: ",r2)
    print("Training Time: ",tr_time," seconds")

plt.figure(figsize = (10,8))
plt.plot(architectures.keys(),rmse_scores,marker = 'o',label = 'RMSE')
plt.plot(architectures.keys(),mae_scores,marker = 's',label = 'MAE')
plt.xlabel("Architecture Layer")
plt.ylabel("Error")
plt.title("RMSE vs MAE")

plt.figure(figsize = (10,8))
plt.bar(architectures.keys(),r2_scores,label = "R2 Score")
plt.xlabel("Architecture Layer")
plt.ylabel("R2 Score")
plt.title("R2 Score Comparison in Different MLP Layers")
plt.show()