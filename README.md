# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load the dataset Salary.csv using pandas and view the first few rows.

2.Check dataset information and identify any missing values.

3.Encode the categorical column "Position" into numerical values using LabelEncoder.

4.Define feature variables x as "Position" and "Level", and target variable y as "Salary".

5.Split the dataset into training and testing sets using an 80-20 split.

6.Create a DecisionTreeRegressor model instance.

7.Train the model using the training data.

8.Predict the salary values using the test data.

9.Evaluate the model using Mean Squared Error (MSE) and R² Score.

10.Use the trained model to predict salary for a new input [5, 6]. 


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Divya R
RegisterNumber:  212222040040
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
## data.head()
![image](https://github.com/user-attachments/assets/5c0b07fe-e1f7-4d34-b1b3-41a986d4d759)
## data.info
![image](https://github.com/user-attachments/assets/b2aed347-5f2e-448a-9c16-7fc4d5d2c37b)
## isnull
![image](https://github.com/user-attachments/assets/c92a80f1-09b2-4c82-9013-cfb78775f660)
## data position
![image](https://github.com/user-attachments/assets/5617d73f-7a4a-46e0-bea2-5f34d399ceb1)
## x.head()
![image](https://github.com/user-attachments/assets/0ac1ba2f-b8de-4ad6-8414-feb860e5922d)
## MSE
![image](https://github.com/user-attachments/assets/6917b25d-080a-477a-9774-47ef0b3ad7d7)
## r2
![image](https://github.com/user-attachments/assets/ec415f5c-4251-4a70-8691-c4278ed72d05)
## Prediction
![image](https://github.com/user-attachments/assets/ddbcf3e7-9406-416e-a388-efca483af755)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
