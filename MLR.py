import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv(r"C:\Users\Lenovo\Desktop\DS&AI\25th April- mlr\MLR\Investment.csv")

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,4]

x = pd.get_dummies(x,dtype=int)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

#== we build mlr model 

m = regressor.coef_
print(m)

c = regressor.intercept_
print(c)

x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1) 


import statsmodels.api as sm
x_opt = x[:,[0,1,2,3,4,5]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()


import statsmodels.api as sm
x_opt = x[:,[0,1,2,3,5]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()


import statsmodels.api as sm
x_opt = x[:,[0,1,2,3]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()


import statsmodels.api as sm
x_opt = x[:,[0,1,3]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()


import statsmodels.api as sm
x_opt = x[:,[0,1]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()


bias = regressor.score(x_train, y_train)
bias

variance = regressor.score(x_test, y_test)
variance
