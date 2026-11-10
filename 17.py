import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = pd.read_csv("digits.csv")
x = data.drop('label',axis = 1)
y = data['label']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

k_vals = [5,8,10,12,15]

inertias = []
silhs = []

for k in k_vals:
    model = KMeans(n_clusters= k,random_state = 42,n_init = 10)

    labels = model.fit_predict(x_scaled)

    inertia = model.inertia_
    silh = silhouette_score(x_scaled,labels)
    inertias.append(inertia)
    silhs.append(silh)
    print(f"Number of Clusters: {k}")
    print("Inertia: ",inertia)
    print("Silhouette Score: ",silh)

plt.figure(figsize=(8,6))
plt.plot(k_vals,inertias,marker ='s')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.figure(figsize=(8,6))
plt.plot(k_vals,silhs,marker = 'o')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Comparison")
plt.show()