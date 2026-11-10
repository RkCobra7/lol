import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv("fashion-mnist_train.csv")
test = pd.read_csv("fashion-mnist_test.csv")

x_train = train.drop('label',axis = 1).values[:5000]
y_train = train['label'].values[:5000]

x_test = test.drop('label',axis = 1).values[:1000]
y_test = test['label'].values[:1000]

x_train = x_train / 255.0
x_test = x_test / 255.0

lrs = [0.001,0.01]
batches = [32,64]
epochs = [20,40]

results = []

for lr in lrs:
    for batch in batches:
        for epoch in epochs:
            model = MLPClassifier(
                hidden_layer_sizes = (128,64),
                activation = 'relu',
                solver = 'adam',
                learning_rate_init = lr,
                batch_size = batch,
                max_iter = epoch,
                random_state = 42
                )
            start = time.time()
            model.fit(x_train,y_train)
            end = time.time()
            tr_time = round(end - start,2)
            y_pred = model.predict(x_test)
            acc = accuracy_score(y_test,y_pred)
            results.append([lr,batch,epoch,acc,tr_time])
result_df = pd.DataFrame(
    results,
    columns = ["Learning Rate","Batch Size","Epochs","Accuracy","Training Time"]
)
print("HyperParameter Comparison")
print(result_df)

plt.figure(figsize = (8,5))
plt.bar(range(len(results)),result_df["Accuracy"])
plt.xlabel("Hyperparameter Combination")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")

plt.figure(figsize = (8,5))
plt.bar(range(len(results)),result_df['Training Time'])
plt.xlabel("Hyperparameter Combination")
plt.ylabel("Training Time")
plt.title("Training Time Comparison")

plt.show()

print("Best HyperParameter Combination")
best = result_df.loc[result_df["Accuracy"].idxmax()]
print(best)