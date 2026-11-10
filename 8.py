import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv("fashion-mnist_train.csv")
test = pd.read_csv("fashion-mnist_test.csv")

x_train = train.drop("label",axis = 1).values
y_train = train['label'].values

x_test = test.drop("label",axis = 1).values
y_test = test['label'].values

x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:2000]
y_test = y_test[:2000]

x_train = x_train / 255.0
x_test = x_test / 255.0

k_vals = [1,3,5,7]
acc = []

for k in k_vals:
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)

    metric = accuracy_score(y_test,y_pred)
    acc.append(metric)

    print("Accuracy for K = ",k," is ",metric)

idx = np.argmax(acc)

print("Best value of k = ",k_vals[idx])
print("Best accuracy = ",acc[idx])

plt.plot(k_vals,acc, marker = 'o')
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.title("KNN performance for multiple K values")
plt.grid()
plt.show()