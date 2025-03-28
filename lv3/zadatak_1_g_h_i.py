import pandas as pd

data = pd.read_csv('data_C02_emission.csv')

#g
vozila_dizel_4cylinders = data[(data['Cylinders'] == 4) & (data['Fuel Type'] == 'D')]
najveca_potrosnja = vozila_dizel_4cylinders.loc[vozila_dizel_4cylinders['Fuel Consumption City (L/100km)'].idxmax()]
print("Vozilo s 4 cilindra i dizel motorom koje ima najveću gradsku potrošnju:")
print(najveca_potrosnja[['Make', 'Model', 'Fuel Consumption City (L/100km)']])

#h
vozila_ručni_mjenjač = data[data['Transmission'] == 'M']
print("Broj vozila s ručnim tipom mjenjača:", len(vozila_ručni_mjenjač))

#i
numeric_data = data.select_dtypes(include='number')
korelacija = numeric_data.corr()
print("Korelacijska matrica između numeričkih veličina:")
print(korelacija)

