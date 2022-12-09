import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#Data collection.
Hours=[2.5,5.1,3.2,8.5,3.5,1.5,9.2,5.5,8.3,2.7,7.7,5.9,4.5,3.3,1.1,8.9,2.5,1.9,6.1,7.4,2.7,4.8,3.8,6.9,7.8]
Scores=[21,47,27,75,30,20,88,60,81,25,85,62,41,42,17,95,30,24,67,69,30,54,35,76,86]

#Making the regression model.
regr = linear_model.LinearRegression()

#Splitting the data into train and test sets by reshaping them into 2d arrays.
train_hr,test_hr,train_scr,test_scr=train_test_split(np.asanyarray(Hours).reshape(len(Hours),1),np.asanyarray(Scores).reshape(len(Scores),1),test_size=0.25,random_state=0)

#Fitting the graph on the data.
regr.fit(train_hr,train_scr)

print ('\nCoefficient: ', regr.coef_[0][0])
print ('Intercept: ',regr.intercept_[0])

#Plotting the fitted graph using matplotlib.
plt.scatter(train_hr,train_scr,color='blue')
plt.plot(train_hr,regr.coef_[0][0]*train_hr+regr.intercept_[0],'-r')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Predicting the values for the test set.
y_pred=regr.predict(test_hr)

#Comparing the test set values with the actual values.
df=pd.DataFrame({'Actual':test_scr.reshape(len(test_scr)),'Predicted':y_pred.reshape(len(y_pred))})
print()
print(df)
print()

#Finding the Mean absolute error, the mean squared error and the R2 Score.
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_pred - test_scr)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_pred - test_scr) ** 2))
print("R2-score: %.2f" % r2_score(test_scr , y_pred) )

#Predicting the value for 9.25 hrs/day.
hr=[[9.25]]
my_pred=regr.predict(hr)
print(f"\nThe score for 9.25 hrs/day is {my_pred[0][0]:.2f}\n")

#Devansh Mathur
#Batch-December 2022
