import matplotlib.pyplot as plt
import numpy as np

img = plt.imread("road.jpg")

brightened_img = img + 0.9

plt.imshow(brightened_img)
plt.title('Posvijetljena slika')
plt.show()
