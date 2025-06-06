import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

dataset = pd.read_csv(r"C:\Users\Lenovo\Desktop\DS&AI\25th April- mlr\MLR\House_data.csv")
dataset.head()

print(dataset.isnull().any())

#checking for categorical data
print(dataset.dtypes)

#dropping the id and date column
dataset = dataset.drop(['id','date'], axis = 1)

#understanding the distribution with seaborn
with sns.plotting_context("notebook",font_scale=2.5):
    g = sns.pairplot(dataset[['sqft_lot','sqft_above','price','sqft_living','bedrooms']], 
                 hue='bedrooms', palette='tab20',height=6)
g.set(xticklabels=[]);
plt.show()

#separating independent and dependent variable
X = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values

#splitting dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Backward Elimination
import numpy as np
import statsmodels.api as sm

def backwardElimination(x, y, SL):
    numVars = x.shape[1]
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues)
        if maxVar > SL:
            maxVar_index = np.argmax(regressor_OLS.pvalues)
            x = np.delete(x, maxVar_index, axis=1)
        else:
            break
    print(regressor_OLS.summary())
    return x

# Example usage:
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17]]  # Make sure X is defined properly
X_Modeled = backwardElimination(X_opt, y, SL)


bias = regressor.score(X_train, y_train)
bias

variance = regressor.score(X_test, y_test)
variance
