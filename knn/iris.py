
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
import numpy as np
import matplotlib.pyplot as plt
import sys
print(sys.version)


iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print('Number of classes: %d' % len(np.unique(iris_y)))
print('Number of data points: %d' % len(iris_y))


X0 = iris_X[iris_y == 0, :]
print('\nSamples from class 0:\n', X0[:5, :])

X1 = iris_X[iris_y == 1, :]
print('\nSamples from class 1:\n', X1[:5, :])

X2 = iris_X[iris_y == 2, :]
print('\nSamples from class 2:\n', X2[:5, :])


# Tach tap du lieu thanh du lieu test va train
X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y, test_size=50)

print("Training size: %d" % len(y_train))
print("Test size    : %d" % len(y_test))

# voi K =1
clf = neighbors.KNeighborsClassifier(n_neighbors=1, p=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Print results for first 20 test data points:")
print("Predicted labels: ", y_pred[20:40])
print("Ground truth    : ", y_test[20:40])


# ham du doan : so luong doan dung/ so luong test
print("Accuracy of 1NN: %.2f %%" % (100*accuracy_score(y_test, y_pred)))

clf = neighbors.KNeighborsClassifier(n_neighbors=9, p=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy of 10NN with major voting: %.2f %%" %
      (100*accuracy_score(y_test, y_pred)))
clf = neighbors.KNeighborsClassifier(n_neighbors=9, p=2, weights='distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy of 10NN (1/distance weights): %.2f %%" %
      (100*accuracy_score(y_test, y_pred)))
