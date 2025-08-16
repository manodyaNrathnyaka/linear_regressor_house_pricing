import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#X is area in square,y is price in $1000,price=50*area+some noise
X=np.array([[650],[800],[950],[1100],[1250],[1400],[1550],[1700],[1850],[2000]])
y=np.array([325,400,475,550,625,700,775.850,925,1000])

model=LinearRegression()
model.fit(X,y)
y_prediction=model.predict(X)
print("co-efficient",model.coef_)
print("intercept",model.intercept_)
print("MSE",mean_squared_error(y,y_prediction))