import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("mnist_784.csv")

x = data.drop('class',axis = 1).values
y = data['class'].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42,stratify = y)

x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:5000]
y_test = y_test[:5000]

x_train = x_train / 255.0
x_test = x_test / 255.0

activations = {
    'Sigmoid' : 'logistic',
    'ReLU' : 'relu',
    'Tanh' : 'tanh'
}

accuracies = []
tr_times = []
iterations = []

for name, func in activations.items():
    model = MLPClassifier(hidden_layer_sizes = (128,64),activation = func,solver ='adam',max_iter = 50,random_state = 42)
    start = time.time()
    model.fit(x_train,y_train)
    end = time.time()
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test,y_pred)
    tr_time = round(end - start,2)
    accuracies.append(acc)
    tr_times.append(tr_time)
    iterations.append(model.n_iter_)
    print("Activation Function: ",name)
    print("Accuracy: ",accuracy_score)
    print("Training Time: ",tr_time," seconds")
    print("Iterations: ",model.n_iter_)

plt.figure(figsize = (10,8))
plt.bar(activations.keys(),accuracies)
plt.xlabel("Activation Function")
plt.ylabel("Accuracy Score")
plt.title("Accuracy Comparison")

plt.figure(figsize = (10,8))
plt.bar(activations.keys(),tr_times)
plt.xlabel("Activation Function")
plt.ylabel("Training Time")
plt.title("Training Time Comparison")

plt.figure(figsize = (10,8))
plt.bar(activations.keys(),iterations)
plt.xlabel("Activation Functions")
plt.ylabel("Iterations")
plt.title("Convergence Comparison")
plt.show()