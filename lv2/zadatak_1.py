import numpy as np
import matplotlib.pyplot as plt

x1 = np.linspace(1.0, 3.0, num = 100)
y1 = np.ones_like(x1) 
plt.plot(x1, y1, 'm', linewidth=2)

y2 = np.linspace(1.0, 2.0, num = 100)
x2 = np.full_like(y2, 3.0)
plt.plot(x2, y2, 'm', linewidth=2)

plt.scatter(3.0, 1.0, color='m', marker='o')

x3 = np.linspace(2.0, 3.0, num = 100)
y3 = np.full_like(x3, 2.0)
plt.plot(x3, y3, 'm', linewidth=2)

plt.scatter(3.0, 2.0, color='m', marker='D')

x4 = [1.0, 2.0]
y4 = [1.0, 2.0]
plt.plot(x4, y4, 'm', linewidth=2)

plt.scatter(1.0, 1.0, color='m', marker='*')
plt.scatter(2.0, 2.0, color='m', marker='^')

plt.axis([0.0, 4.0, 0.0, 4.0])
plt.xlabel('x os')
plt.ylabel('y os')
plt.title('Primjer')
plt.show()