import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train = pd.read_csv("fashion-mnist_train.csv")
test = pd.read_csv("fashion-mnist_test.csv")

x_train = train.drop('label',axis = 1).values
y_train = train['label'].values

x_test = test.drop('label',axis = 1).values
y_test = test['label'].values

x_train = x_train[:5000]
y_train = y_train[:5000]
x_test = x_test[:1000]
y_test = y_test[:1000]
x_train = x_train/255.0
x_test = x_test/255.0

models = {
    'Linear' : SVC(kernel = 'linear',C=1),
    'Polynomial' : SVC(kernel = 'poly',degree = 3,C = 1,gamma = 'scale'),
    'RBF' : SVC(kernel = 'rbf',C = 1,gamma = 'scale')
}

training_times = []
accuracies = []
for name,model in models.items():
    start = time.time()
    model.fit(x_train,y_train)
    end = time.time()
    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test,y_pred)
    training_time = end - start
    training_times.append(training_time)
    accuracies.append(acc)
    print(f"{name} Kernel")
    print("Accuracy: ",acc)
    print(f"Training Time: {training_time} seconds")

plt.figure(figsize = (10,8))
plt.bar(models.keys(),accuracies)
plt.xlabel("SVM Kernel")
plt.ylabel("Accuracy Score")
plt.title("Accuracy Comparison")

plt.figure(figsize = (10,8))
plt.bar(models.keys(),training_times)
plt.xlabel("SVM Models")
plt.ylabel("Training Time")
plt.title("Training Time Comparison")
plt.show()