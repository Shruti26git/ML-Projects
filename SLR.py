import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\Lenovo\Desktop\DS&AI\18th April- ML\5. Data preprocessing\Data.csv")
df


X = df.iloc[:, :-1].values	
y = df.iloc[:,3].values  


from sklearn.impute import SimpleImputer
imputer = SimpleImputer() 


imputer = imputer.fit(X[:,1:3]) 
X[:, 1:3] = imputer.transform(X[:,1:3])


from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
labelencoder_X.fit_transform(X[:,0]) 
X[:,0] = labelencoder_X.fit_transform(X[:,0]) 


labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size= 0.2, random_state=0)


from sklearn.preprocessing import Normalizer
sc_X = Normalizer() 
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)