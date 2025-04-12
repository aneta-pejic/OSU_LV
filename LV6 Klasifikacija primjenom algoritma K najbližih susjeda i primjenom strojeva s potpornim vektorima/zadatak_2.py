import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

data = pd.read_csv("Social_Network_Ads.csv")
X = data[['Age', 'EstimatedSalary']].values
y = data['Purchased'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

k_range = range(1, 101)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_k = k_range[np.argmax(cv_scores)]
print("Optimalna vrijednost K:", optimal_k)

plt.figure()
plt.plot(k_range, cv_scores, marker='*')
plt.title('Točnost modela za različite vrijednosti K (unakrsna validacija)')
plt.xlabel('Broj susjeda K')
plt.ylabel('Točnost')
plt.show()