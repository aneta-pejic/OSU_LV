import matplotlib.pyplot as plt
import numpy as np

img = plt.imread("road.jpg")

height, width, _ = img.shape

b1 = 0
b2 = height 
a1 = width / 4 
a2 = width / 2  

cropped_image = img[b1:b2, a1:a2]

plt.imshow(cropped_image)
plt.title('Druga četvrtina slike po širini')
plt.show()
