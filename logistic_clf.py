import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split


n_samples = 10000

centers = [(-1, -1), (1, 1)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.8,
                  centers=centers, shuffle=False, random_state=19)

y[:n_samples // 2] = 0
y[n_samples // 2:] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=19)
log_reg = linear_model.LogisticRegression()



# some code here ( fit and prediction)

log_reg.fit(X_train, y_train)
predictions = log_reg.predict(X_test)

#print(np.unique(predictions))   #       shows that inside the numpy array there are only 2 sets: 0 and 1

plt.scatter(X_test[:,0], X_test[:,1], c=predictions)
plt.show()