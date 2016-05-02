import numpy as np
import matplotlib.pyplot as plt

# Define class for adaptive linear neuron object

class AdalineGD(object):
    """ ADAptive LInear NEuron Classifier.
    
    Parameters
    -------------
    eta : float
        learning rate, between 0.0 and 1.0
    
    n_iter : int
        passes over the training set
    
    Attributes
    ------------
    w_ : 1d-array
        weights after fitting
    errors_ : list
        number of misclassifications in every epoch
    
    """
    
    def __init__(self, eta = 0.01, n_iter = 50):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        """ Fit training data.
        
        Parameters
        ------------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors,
            where n_samples is the number of samples and n_features is
            the number of features
        y : array-like, shape = [n_samples]
            Target values
        
        Returns
        ------------
        self : object
        """
        
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
        
    def net_input(self, X):
        """ calculates net input """
        return np.dot(X, self.w_[1:]) + self.w_[0]
        
    def activation(self, X):
        """ compute linear activation """
        return self.net_input(X)
        
    def predict(self, X):
        """ return class label after unit step """
        return np.where(self.activation(X) >= 0.0, 1, -1)
        
        
# function to visualize decision boundaries for 2d
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution = 0.02):
    
    # set up marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1 # determine min and max of graph axes
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # create grid based on mins and maxes
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                           np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1],
                    alpha = 0.8, c = cmap(idx),
                    marker = markers[idx], label = cl)
                    
                    
                    
# import data for classification
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa', 1, -1)
X = df.iloc[0:100, [0,2]].values


# Plot cost against number of epochs for two different learning rates
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize =(8,4))
ada1 = AdalineGD(n_iter = 500, eta = 0.01).fit(X,y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker = 'o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning Rate 0.01')

ada2 = AdalineGD(n_iter = 500, eta = 0.0001).fit(X,y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker = 'o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning Rate 0.0001')

plt.show()

# standardize features
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

# fit adaline model on standardized data and plot decision boundary
ada = AdalineGD(n_iter = 25, eta = 0.01)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier = ada)
plt.title("Adaline - Gradient Descent")
plt.xlabel('Sepal Length (standardized)')
plt.ylabel('Petal Length (standardized)')
plt.legend(loc = 'upper left')
plt.show()

plt.plot(range(1,len(ada.cost_) + 1), ada.cost_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Sum Squared Error')
plt.show()



