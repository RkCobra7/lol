import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

data = pd.read_csv("adult.csv")
data = data.dropna()
x = data.drop('income',axis = 1)
y = data['income'].map({"<=50K": 0,">50K": 1})

x = pd.get_dummies(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

lr = LogisticRegression(max_iter = 10000)
lr.fit(x_train_scaled,y_train)
lr_pred = lr.predict(x_test_scaled)
print("Logistic Regression")
print("Accuracy: ",accuracy_score(y_test,lr_pred))
print("Precision: ",precision_score(y_test,lr_pred))
print("Recall: ",recall_score(y_test,lr_pred))
print("F1 Score: ",f1_score(y_test,lr_pred))

dt = DecisionTreeClassifier(criterion = 'entropy',max_depth = 5,random_state = 42)
dt.fit(x_train,y_train)
dt_pred = dt.predict(x_test)
print("Decision Tree")
print("Accuracy: ",accuracy_score(y_test,dt_pred))
print("Precision: ",precision_score(y_test,dt_pred))
print("Recall: ",recall_score(y_test,dt_pred))
print("F1 Score: ",f1_score(y_test,dt_pred))

metrics = ["Accuracy","Precision","Recall","F1 Score"]
lr_scores = [accuracy_score(y_test,lr_pred),precision_score(y_test,lr_pred),recall_score(y_test,lr_pred),f1_score(y_test,lr_pred)]
dt_scores = [accuracy_score(y_test,dt_pred),precision_score(y_test,dt_pred),recall_score(y_test,dt_pred),f1_score(y_test,dt_pred)]

plt.figure(figsize = (10,8))

x_idx = np.arange(len(metrics))
width = 0.35

plt.bar(x_idx - width/2,lr_scores,width,label = "Logistic Regression")
plt.bar(x_idx + width/2,dt_scores,width,label = "Decision Tree")

plt.xticks(x_idx,metrics)
plt.xlabel("Metrics")
plt.ylabel("Score")
plt.title("Logistic Regression vs Decision Tree")
plt.legend()
plt.tight_layout()
plt.show()