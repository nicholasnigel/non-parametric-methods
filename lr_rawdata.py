import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import seaborn
seaborn.set()

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

train, test = train_test_split(df, test_size = 0.2, shuffle=False)      #       data splitting into train and tests

scaler = StandardScaler()

train_sc = scaler.fit_transform(train[['horsepower','price']])

test_sc = scaler.transform(test[['horsepower','price']])

# Regression part
regressor = LinearRegression()

regressor.fit(np.asarray(train_sc[:,0].reshape(-1,1)), np.asarray(train_sc[:,1]))
predictions = regressor.predict(test_sc[:,0].reshape(-1,1))

print(predictions)

# ================================  Plotting    ================================

#plt.plot(test_sc[:,1],predictions,'ro')
plt.plot(test_sc[:,0], predictions, 'ro')
plt.plot(test_sc[:,0], test_sc[:,1],'bo')

plt.show()
