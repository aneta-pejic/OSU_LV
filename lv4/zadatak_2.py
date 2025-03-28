import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('data_C02_emission.csv')

numerical_columns = ['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Type']
X = data[numerical_columns] 
y = data['CO2 Emissions (g/km)'] 

ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(data[['Fuel Type']]).toarray()
print(numerical_columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = LinearRegression()

y_pred = model.predict(X_test)

error = abs(y_pred - y_test)
max_error_index = error.idxmax() 
max_error_value = error.max()

print("Maksimalna pogreška u procjeni emisije CO2:", max_error_value, "g/km")
print("Model vozila s maksimalnom pogreškom:", data.iloc[max_error_index]['Model'])


