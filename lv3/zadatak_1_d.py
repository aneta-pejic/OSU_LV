import pandas as pd

data = pd.read_csv('data_C02_emission.csv')

audi_vozila = data[data['Make'] == 'Audi']
broj_audi_vozila = len(audi_vozila)
print("Broj Audi vozila:", broj_audi_vozila)

audi_4cylinders = audi_vozila[audi_vozila['Cylinders'] == 4]
prosjecna_emisija = audi_4cylinders['CO2 Emissions (g/km)'].mean()
print("Prosječna emisija CO2 za Audi sa 4 cilindara:", prosjecna_emisija, "g/km")