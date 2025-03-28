import pandas as pd

data = pd.read_csv('data_C02_emission.csv')

print("DataFrame sadrži", len(data), "mjerenja.")

print("\nTipovi veličina:\n", data.dtypes)

print("\nIzostale vrijednosti\n", data.isnull().sum())
data = data.dropna()

print("\nBroj dupliciranih redaka:\n", data.duplicated().sum())
data = data.drop_duplicates()

for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].astype('category')
