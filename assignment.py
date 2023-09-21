import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv("SwedishMotorInsurance.csv")
dataset
x=dataset.iloc[:, :-1]
y=dataset.iloc[:, -1]
print(x)
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state=1)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(x_train, y_train)
y_pred = lasso_model.predict(x_val)
mse = mean_squared_error(y_val, y_pred)
print(f"Mean Squared Error on Validation Set: {mse}")
coefficients = lasso_model.coef_
print("Coefficients of the selected features:")
print(coefficients)

y_pred = lasso_model.predict(x_test)
print(y_pred)

dataset = pd.read_csv("SwedishMotorInsurance.csv")
x = dataset['Insured']
y = dataset['Claims']
plt.figure(figsize=(20, 20))
plt.scatter(x, y, color='blue', marker='o', label='Scatterplot')
plt.xlabel('Insured')
plt.ylabel('Claims')
plt.title('Scatterplot for insured to claims')
plt.show()
