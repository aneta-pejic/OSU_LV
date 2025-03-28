import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('data_C02_emission.csv')

#a
numerical_columns = ['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 
                     'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)']
X = data[numerical_columns] 
y = data['CO2 Emissions (g/km)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#print("Broj podataka u skupu za treniranje:", len(X_train))
#print("Broj podataka u skupu za testiranje:", len(X_test))


#b
plt.figure()
plt.scatter(X_train['Engine Size (L)'], y_train, color='blue', label='Trening skup')
plt.scatter(X_test['Engine Size (L)'], y_test, color='red', label='Testni skup')
plt.title('Ovisnost emisije CO2 o veličini motora')
plt.legend()
plt.show()

#c
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
plt.figure()
plt.hist(X_train['Engine Size (L)'], color="purple", edgecolor="black")
plt.title('Histogram veličine motora prije standardizacije')
plt.show()

plt.figure()
plt.hist(X_train_scaled[:, 0], edgecolor="black")
plt.title('Histogram veličine motora nakon standardizacije')
plt.show()

#d
model = lm.LinearRegression()
model.fit(X_train_scaled, y_train)
print("\nParametri modela (koeficijenti):")
print(model.coef_) 
print("Intercept:")
print(model.intercept_)

#e
X_test_scaled = scaler.fit_transform(X_test)
y_pred = model.predict(X_test_scaled)
plt.figure()
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', lw=2)
plt.title('STVARNE vs PREDVIĐENE vrijednosti emisije CO2')
plt.show()

#f
mse = mean_squared_error(y_test, y_pred) 
rmse = np.sqrt(mse) 
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nmean squared error (MSE):", mse)
print("root mean squared error (RMSE):", rmse)
print("koeficijent determinacije (R2):", r2)
print("mean absolute percentage error (MAPE):", mape)
print("mean absolute error:", mae)

#g
X_train_reduced = X_train[['Engine Size (L)', 'Cylinders']]  
X_test_reduced = X_test[['Engine Size (L)', 'Cylinders']]

X_train_reduced_scaled = scaler.fit_transform(X_train_reduced)
X_test_reduced_scaled = scaler.transform(X_test_reduced)

model_reduced = LinearRegression()
model_reduced.fit(X_train_reduced_scaled, y_train)

y_pred_reduced = model_reduced.predict(X_test_reduced_scaled)

mse_reduced = mean_squared_error(y_test, y_pred_reduced)
rmse_reduced = np.sqrt(mse_reduced) 
r2_reduced = r2_score(y_test, y_pred_reduced)
mape_reduced = mean_absolute_percentage_error(y_test, y_pred_reduced)
mae_reduced = mean_absolute_error(y_test, y_pred_reduced)

print("\nEvaluacija modela s manjim brojem ulaznih varijabli:")
print("MSE:", mse_reduced)
print("RMSE:", rmse_reduced)
print("R2:", r2_reduced)
print("MAPE", mape)
print("MAE:", mae)