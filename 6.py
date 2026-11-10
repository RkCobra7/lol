import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score

data = pd.read_csv("pima_indians_diabetes.csv")

x = data.drop("Outcome",axis = 1)
y = data['Outcome']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

model1 = LogisticRegression(max_iter = 1000)
model1.fit(x_train,y_train)
y_pred1 = model1.predict(x_test)

print("Without Scaling")
print("Accuracy: ",accuracy_score(y_test,y_pred1))
print("Precision: ",precision_score(y_test,y_pred1))
print("Recall: ",recall_score(y_test,y_pred1))
print("F1 Score: ",f1_score(y_test,y_pred1))

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
model2 = LogisticRegression(max_iter = 1000)
model2.fit(x_train_scaled,y_train)
y_pred2 = model2.predict(x_test_scaled)

print("With Scaling")
print("Accuracy: ",accuracy_score(y_test,y_pred2))
print("Precision: ",precision_score(y_test,y_pred2))
print("Recall: ",recall_score(y_test,y_pred2))
print("F1 Score: ",f1_score(y_test,y_pred2))