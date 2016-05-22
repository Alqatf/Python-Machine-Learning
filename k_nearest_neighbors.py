# Python Machine Learning
# K Nearest Neighbors

# prepare data
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
# split data to evaluate model performance
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# normalize features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# combine train/test for visualization
import numpy as np
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# fit knn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5,
                           p = 2, # p = 2 makes the general minkowski distance the same as euclidean, p = 1 is Manhattan distance
                           metric = 'minkowski')

knn.fit(X_train_std, y_train)

# plot decision regions
import sys
import matplotlib.pyplot as plt
sys.path.append("C:/Users/Craig/Documents/GitHub/Python-Machine-Learning")
from functions_module import plot_decision_regions

plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier = knn,
                      test_idx = range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.show()
