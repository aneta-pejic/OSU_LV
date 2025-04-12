import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("Social_Network_Ads.csv")

X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

svm_rbf = SVC(kernel='rbf', C=1.0, gamma=0.1)  
svm_rbf.fit(X_train, y_train)

y_pred = svm_rbf.predict(X_test)
print("Točnost na testnim podacima:", {accuracy_score(y_test, y_pred)})

def plot_decision_boundary(X, y, model, ax):
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.75)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor='k')
    return scatter

fig, ax = plt.subplots()

svm_rbf = SVC(kernel='rbf', C=10.0, gamma=0.01) 
svm_rbf.fit(X_train, y_train)

y_pred = svm_rbf.predict(X_test)
print("Točnost na testnim podacima (C=10, γ=0.01):", {accuracy_score(y_test, y_pred)})

fig, ax = plt.subplots()
plot_decision_boundary(X_train, y_train, svm_rbf, ax)
ax.set_title('SVM with RBF kernel (C=10.0, γ=0.01)')
plt.show()