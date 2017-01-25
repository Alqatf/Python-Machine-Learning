# Python Machine Learning
# Support Vector Machines

# prepare data
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

# split data to evaluate model performance
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# normalize features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# train SVM
from sklearn.svm import SVC
svm = SVC(kernel = 'linear', C = 1.0, random_state = 0)
svm.fit(X_train_std, y_train)

# plot decision regions of SVM
import sys
import numpy as np
sys.path.append("~/Documents/Practice/Repos/Python-Machine-Learning")
from functions_module import plot_decision_regions

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier = svm,
                      test_idx = range(105, 150))


# kernel SVM

# create XOR gate sample data
import matplotlib.pyplot as plt

np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c = 'b',
            marker = 'x',
            label = '1')
            
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c = 'r',
            marker = 's',
            label = '-1')
            
plt.ylim(-3.0)
plt.legend()
plt.show()

# fit SVM using radial basis function kernel
svm = SVC(kernel = 'rbf',
          random_state = 0,
          gamma = 0.10,
          C = 10.0)

svm.fit(X_xor, y_xor)

# plot SVM with fitted decision regions
plot_decision_regions(X_xor,
                      y_xor,
                      classifier = svm)

plt.legend(loc = 'upper left')
plt.show()

# RBF SVM on Iris data
svm = SVC(kernel = 'rbf',
          random_state = 0,
          gamma = 0.2,
          C = 1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier = svm,
                      test_idx = range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc = 'upper left')
plt.show()


# try with a larger value of gamma
svm = SVC(kernel = 'rbf',
          random_state = 0,
          gamma = 100.0,
          C = 1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier = svm,
                      test_idx = range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc = 'upper left')
plt.show()
