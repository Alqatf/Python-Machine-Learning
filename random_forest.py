# Python Machine Learning
# Random Forests

# prepare data
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

# split data to evaluate model performance
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion = 'entropy',
                                n_estimators = 10,
                                random_state = 1,
                                n_jobs = 2)
forest.fit(X_train, y_train)

# Plot decision regions of model
import numpy as np
import matplotlib.pyplot as plt

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

import sys
sys.path.append("C:/Users/Craig/Documents/GitHub/Python-Machine-Learning")
from functions_module import plot_decision_regions
plot_decision_regions(X_combined,
                      y_combined,
                      classifier = forest,
                      test_idx = range(105, 150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc = 'upper left')
plt.show()
