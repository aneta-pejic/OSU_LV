import pandas as pd

data = pd.read_csv('data_C02_emission.csv')

filtered_data = data[(data['Engine Size (L)'] > 2.5) & (data['Engine Size (L)'] < 3.5)]
motor_size_count = len(filtered_data)

average_co2 = filtered_data['CO2 Emissions (g/km)'].mean()

print('Broj vozila koji imaju veličinu motora između 2.5 i 3.5 L:', motor_size_count)
print('Prosječna CO2 emisija za ta vozila:', average_co2, 'g/km')
