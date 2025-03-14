import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)

visina = data[:, 1]
masa = data[:, 2]   

visina_50 = visina[::50]  
masa_50 = masa[::50]    

plt.scatter(visina_50, masa_50, color='m', marker='o')  

plt.xlabel('visina (cm)')
plt.ylabel('masa (kg)')
plt.title('Odnos visine i mase (svaka pedeseta osoba)')

plt.show()
