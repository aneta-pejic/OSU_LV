import pandas as pd

data = pd.read_csv('data_C02_emission.csv')

print("\nTri automobila s najvećom gradskom potrošnjom:")
najveca_potrosnja = data.sort_values(by='Fuel Consumption City (L/100km)').tail(3)
print(najveca_potrosnja[['Make', 'Model', 'Fuel Consumption City (L/100km)']])

print("\nTri automobila s najmanjom gradskom potrošnjom:")
najmanja_potrosnja = data.sort_values(by='Fuel Consumption City (L/100km)').head(3)
print(najmanja_potrosnja[['Make', 'Model', 'Fuel Consumption City (L/100km)']])
