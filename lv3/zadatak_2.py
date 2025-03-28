import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data_C02_emission.csv')

#a
plt.figure()
data['CO2 Emissions (g/km)'].plot(kind='hist', bins = 20, edgecolor="black")
plt.title("Emisija CO2 plinova")
plt.show()

#b
plt.figure()

colors = {'X': 'orange', 'D': 'purple', 'Z': 'green', 'E': 'blue', 'N': 'red'}
color_list = [colors[group] for group in data['Fuel Type']]

data.plot.scatter(x='Fuel Consumption City (L/100km)', y='CO2 Emissions (g/km)', c=color_list)
plt.show()

#c
plt.figure()
data.boxplot(column='Fuel Consumption Hwy (L/100km)', by='Fuel Type')
plt.title("Izvangradska potrošnja po tipu goriva")
plt.show()

#d
plt.figure()
fuel_counts = data.groupby('Fuel Type').size()
fuel_counts.plot(kind='bar', color='blue')
plt.title("Broj vozila po tipu goriva")
plt.show()

#e
plt.figure()
cyl_avg_co2 = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
cyl_avg_co2.plot(kind='bar', color='green')
plt.title("Prosječna CO2 emisija po broju cilindara")
plt.show()
