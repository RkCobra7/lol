import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = pd.read_csv("Iris.csv")
data = data.drop("Id",axis = 1)

x = data[['SepalLengthCm','SepalWidthCm']].values
y = (data['Species'] == 'Iris-setosa').astype(int).values

svm = SVC(kernel = 'linear',C = 1)
svm.fit(x,y)
pred = svm.predict(x)
print("Accuracy: ",accuracy_score(y,pred))

w = svm.coef_[0]
b = svm.intercept_[0]

x_values = np.array([x[:,0].min(),x[:,0].max()])

# w0x + w1y + b = 0
# w0x + w1y + b = 1
# w0x + w1y + b = -1

y_values = -(w[0] * x_values + b) / w[1]
y_upper = -(w[0] * x_values + b - 1) / w[1]
y_lower = -(w[0] * x_values + b + 1) / w[1]

plt.scatter(x[y==1,0],x[y==1,1],label = "Setosa")
plt.scatter(x[y==0,0],x[y==0,1],label = "Non-Setosa")

plt.plot(x_values,y_values,label = "Decision Boundary")
plt.plot(x_values,y_upper,"--",label = "Upper Margin")
plt.plot(x_values,y_lower,"--",label = "Lower Margin")

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Linear SVM - Iris Dataset")
plt.legend()
plt.show()