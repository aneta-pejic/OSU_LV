import pandas as pd

data = pd.read_csv('data_C02_emission.csv')

vehicles_468cylinders = data[data['Cylinders'].isin([4, 6, 8])]
broj_vozila = len(vehicles_468cylinders)
print("Broj vozila s 4, 6 ili 8 cilindara:", broj_vozila)

prosjecna_emisija = vehicles_468cylinders['CO2 Emissions (g/km)'].mean()
print("Prosječna emisija CO2 za vozila sa 4, 6 ili 8 cilindara:", prosjecna_emisija, "g/km")