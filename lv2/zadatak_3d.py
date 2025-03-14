import matplotlib.pyplot as plt
import numpy as np

img = plt.imread("road.jpg")

mirrored_image = np.flip(img, axis=1)

plt.imshow(mirrored_image)
plt.title('Zrcaljena slika')
plt.show()
