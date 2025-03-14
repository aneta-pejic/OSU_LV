import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data.csv', delimiter = ',', skip_header = 1)

#a
broj_osoba = len(data)
print("Mjerenja su izvršena na", broj_osoba, "osoba.")

#b
visina = data[:, 1]
masa = data[:, 2]    

plt.scatter(visina, masa, color='m', marker='o')  

plt.xlabel('visina (cm)')
plt.ylabel('masa (kg)')
plt.title('Odnos visine i mase osoba')

plt.show()

#d
print("Minimalna visina:", visina.min(), "cm.")
print("Maximalna visina:", visina.max(), "cm.")
print("Prosječna visina:", visina.mean(), "cm.")
