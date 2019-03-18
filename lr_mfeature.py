import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn
seaborn.set()

# ====================================  Linear Regression with Mulitple Feature ============================================#

# Features used: city-mpg, horsepower, engine-size, peak-rpm
column_heads = [
    'symboling','normalized-losses','make','fuel-type','aspiration','num-of-doors','body-style',
    'drive-wheels','engine-location','wheel-base','length', 'width', 'height', 'curb-weight','engine-type','num-of-cylinders',
    'engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price'
    ]

df = pd.read_csv('imports-85.data',
            header=None,
            names= column_heads,
            na_values=('?')
                )

df = df.dropna()        #       Drop rows with NaN values
print(df.shape)
#train, test = train_test_split(df, test_size = 0.2, shuffle=False)      #       data splitting into train and 

scaler = StandardScaler()       #        standard scaler


data_scaled = scaler.fit_transform(df[['city-mpg','horsepower','engine-size','peak-rpm','price']])


# ========================================  Split into X and y _train and _test ============================================

# Data Sampling
X = data_scaled[:,:4]
y = data_scaled[:,4]

# ========================================  Using Normal Equation ================================

# adding unit feature into the X -> X_new

ones = np.ones(data_scaled.shape[0])
ones = ones.reshape(-1,1)
X_new = np.concatenate((ones,X),axis=1)
#print(X_new)

#  Solving theta

#   Divide into 2 parts : the inverse and the other one

# left part: (X^T X)^-1
left = np.dot(np.transpose(X_new), X_new)
left = np.linalg.inv(left)

# right part: X^T y
right = np.dot(np.transpose(X_new), y)

theta =  np.dot(left,right)
# From normal equation you get:
print(theta)


# ========================================  Multiple feature wiith gradient descent ====================================
regressor = linear_model.SGDRegressor(loss='squared_loss')

regressor.fit(X,y)

print(regressor.intercept_)
print(regressor.coef_)