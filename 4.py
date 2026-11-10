import numpy as np
import pandas as pd
import matplotlib.pyplot

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

data = pd.read_csv("breast_cancer.csv")
data = data.drop(columns = ['id'],errors = 'ignore')

x = data.drop('diagnosis',axis = 1)
y = data['diagnosis'].map({'M':1,'B':0})

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42,stratify = y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

mle = LogisticRegression(penalty = None,max_iter = 10000)
mle.fit(x_train,y_train)
mle_pred = mle.predict(x_test)

C = [0.01,0.1,1,10,100]

l1 = GridSearchCV(LogisticRegression(penalty ='l1',solver = 'liblinear',max_iter = 10000),{'C':C},cv = 5)
l1.fit(x_train,y_train)
l1_pred = l1.predict(x_test)

l2 = GridSearchCV(LogisticRegression(penalty = 'l2',max_iter = 10000),{'C':C},cv = 5)
l2.fit(x_train,y_train)
l2_pred = l2.predict(x_test)

models = ['MLE','MAP-L1','MAP-L2']
preds = [mle_pred,l1_pred,l2_pred]
acc_scores = [accuracy_score(y_test,p) for p in preds]
pre_scores = [precision_score(y_test,p) for p in preds]
rec_scores = [recall_score(y_test,p) for p in preds]
f1_scores = [f1_score(y_test,p) for p in preds]
print("\t\tMLE\tMAP-L1\tMAP-L2\t")
print(f"Accuracy: {acc_scores[0]}\t{acc_scores[1]}\t{acc_scores[2]}")
print(f"Precision: {pre_scores[0]}\t{acc_scores[1]}\t{pre_scores[2]}")
print(f"Recall: {rec_scores[0]}\t{acc_scores[1]}\t{rec_scores[2]}")
print(f"F1 Score: {f1_scores[0]}\t{acc_scores[1]}\t{f1_scores[2]}")
print("Best C values:")
print("L1: ",l1.best_params_['C'])
print("L2: ",l2.best_params_['C'])
print("Coefficient: ")