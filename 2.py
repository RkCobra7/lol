import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

data = pd.read_csv("auto-mpg.csv")

data = data.dropna(subset = ['displacement','mpg'])

x = data[['displacement']].values
y = data['mpg'].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

degrees = [1,2,3,4]

x_curve = np.linspace(x.min(),x.max(),300).reshape(-1,1)

plt.figure(figsize = (10,6))

plt.scatter(x_test,y_test,color = "black",label = "Actual data")

for d in degrees:

    poly = PolynomialFeatures(degree = d)

    x_train_poly = poly.fit_transform(x_train)
    x_curve_poly = poly.transform(x_curve)

    model = LinearRegression()
    model.fit(x_train_poly,y_train)
    y_curve = model.predict(x_curve_poly)
    plt.plot(x_curve,y_curve,color = "red",label = f'Degree {d}')

    x_test_poly = poly.transform(x_test)
    y_pred = model.predict(x_test_poly)
    print("Degree ",d)
    print("Mean Squared Error: ",mean_squared_error(y_test,y_pred))
    print("R Squared: ",r2_score(y_test,y_pred))

plt.xlabel("Displacement")
plt.ylabel("MPG")
plt.title("Polynomial Regression for different degrees")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

