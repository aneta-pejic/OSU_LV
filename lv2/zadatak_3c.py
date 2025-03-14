import matplotlib.pyplot as plt
import numpy as np

img = plt.imread("road.jpg")

rotated_image = np.rot90(img, k=3)  

plt.imshow(rotated_image)
plt.title('Slika rotirana za 90 stupnjeva u smjeru kazaljke na satu')
plt.show()



