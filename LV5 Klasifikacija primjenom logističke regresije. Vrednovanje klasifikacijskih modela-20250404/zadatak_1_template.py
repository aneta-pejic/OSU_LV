import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


#a
plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='Pastel2', label='Train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='Pastel2', marker='x', label='Test')
plt.legend()
plt.show()

#b
model = LogisticRegression()
model.fit(X_train, y_train)

#c
theta0 = model.intercept_[0]
theta1, theta2 = model.coef_[0]

x = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y = -(theta1 * x + theta0) / theta2

plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='prism', label='Train')
plt.plot(x, y, 'k-', label='Granica odluke')
plt.legend()
plt.title("Granica odluke logističke regresije")
plt.show()

#d
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Matrica zabune:\n", cm)
print("Točnost:", accuracy)
print("Preciznost:", precision)
print("Odziv:", recall)

#e
correct = y_pred == y_test
colors = np.where(correct, 'green', 'black')

plt.figure()
plt.scatter(X_test[:, 0], X_test[:, 1], c=colors)
plt.title("Klasifikacija testnih podataka (zeleno: točno, crno: netočno)")
plt.show()
