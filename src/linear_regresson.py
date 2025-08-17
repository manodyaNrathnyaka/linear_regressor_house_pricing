import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split


#X is area in square,y is price in $1000,price=50*area+some noise
#X=np.array([[650],[800],[950],[1100],[1250],[1400],[1550],[1700],[1850],[2000]])
#y=np.array([325,400,475,550,625,700,775,850,925,1000])
data=pd.read_csv("../data/Boston.csv")
# print(data.head())
# print(data.info())#show data types and not-null counts
# #print(data.describe())#summary statistics
# print(data.isnull().sum())

# sns.heatmap(data.isnull(), cbar=True)
# plt.show()

# sns.boxplot(x=data['medv'])
# plt.show()

Q1=data['medv'].quantile(0.25)
Q3=data['medv'].quantile(0.75)
IQR=Q3-Q1

data=data[(data['medv']>=Q1-1.5*IQR)&(data['medv']<=Q3+1.5*IQR)]
# sns.boxplot(x=data['medv'])
# plt.show()
# for col in data.columns:
#     if col != 'medv':
#         plt.scatter(data[col],data['medv'])
#         plt.xlabel(col)
#         plt.ylabel('medv')
#         plt.title(f'{col} vs medv')
#         plt.show()
        
# corr_matrix = data.corr()
# print(corr_matrix)
# sns.heatmap(corr_matrix, annot=True)
# plt.show()
#model=LinearRegression()
#model.fit(X,y)
#y_prediction=model.predict(X)
#print("co-efficient",model.coef_)
#print("intercept",model.intercept_)
#print("MSE",mean_squared_error(y,y_prediction))

X=data[['rm','black','zn','dis']]
y=data['medv']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

new_data=pd.DataFrame([[6.5,350.0,0.0,4.0]],columns=['rm','black','zn','dis'])
predicted_price=model.predict(new_data)
print("predicted price",predicted_price[0])

# plt.scatter(X, y, color='blue', label='Actual Data')
# plt.plot(X, y_prediction, color='red', label='Regression Line')
# plt.scatter(new_area, predicted_price, color='green', label='Prediction (1600 sq.ft)')
# plt.xlabel("Area (sq.ft)")
# plt.ylabel("Price ($1000s)")
# plt.title("House Price Prediction using Linear Regression")
# plt.legend()
# plt.grid(True)
# plt.show()