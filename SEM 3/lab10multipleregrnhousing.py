# -*- coding: utf-8 -*-
"""LAB10multipleregrnhousing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yKYfN4bUNFOM18dWMkXgGWbMSOnZYfh3
"""



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plot

data=pd.read_csv('/content/drive/MyDrive/dataset/Housing - Housing.csv')

print(data)

data.info()

data.describe()

data.isnull().sum()

#binary categorical columns  converted to numerical values
def toNumeric(x):
    return x.map({"no":0,"yes":1})
def convert_binary():
    for column in list(data.select_dtypes(['object']).columns):
        if(column != 'furnishingstatus'):
            data[[column]] = data[[column]].apply(toNumeric)
convert_binary()

status = pd.get_dummies(data['furnishingstatus'])
status

status = pd.get_dummies(data['furnishingstatus'], drop_first=True)

data = pd.concat([data, status], axis=1)

data.drop(columns='furnishingstatus',inplace=True)

# Split data into features (X) and target variable (y)
X = data.drop(columns=['price'])  # Use all columns except 'price' as features
y = data['price']

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model=LinearRegression()

model.fit(X_train, y_train)

y_pred=model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = (mse ** 0.5)
r2 = r2_score(y_test, y_pred)

# Print the model's coefficients and evaluation metrics
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared (R2) Score:", r2)

plot.scatter(y_test,y_pred)
plot.title('y_test vs y_pred', fontsize=20)
plot.xlabel('y_test', fontsize=18)
plot.ylabel('y_pred', fontsize=16)