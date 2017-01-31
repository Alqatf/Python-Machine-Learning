# Python Machine Learning
# Random Forests

# prepare data
from sklearn import datasets
boston = datasets.load_boston()
X = boston.data
y = boston.target
feature_names = boston.feature_names

# split data to evaluate model performance
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators = 500,
                               criterion = 'mse',
                               max_features='sqrt',
                               oob_score=True,
                               verbose=1)

forest.fit(X_train, y_train)

forest.feature_importances_

forest.predict(X_test)

forest.oob_score_
