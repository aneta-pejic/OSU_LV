import pandas as pd

data = pd.read_csv('data_C02_emission.csv')

dizel_vozila = data[data['Fuel Type'] == 'D']
regular_vozila = data[data['Fuel Type'] == 'Z']

diesel_avg = dizel_vozila['Fuel Consumption City (L/100km)'].mean()
regular_avg = regular_vozila['Fuel Consumption City (L/100km)'].mean()

diesel_median = dizel_vozila['Fuel Consumption City (L/100km)'].median()
regular_median = regular_vozila['Fuel Consumption City (L/100km)'].median()

print("Prosječna gradska potrošnja (Dizel):", diesel_avg, "L/100km")
print("Prosječna gradska potrošnja (Regular):", regular_avg, "L/100km")
print("Medijalna gradska potrošnja (Dizel):", diesel_median, "L/100km")
print("Medijalna gradska potrošnja (Regular):", regular_median, "L/100km")
