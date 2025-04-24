import numpy as np
import tensorflow as tf
import keras
from PIL import Image
import matplotlib.pyplot as plt

model = keras.models.load_model("mnist_model.h5")

image_path = "broj2.png"
img = Image.open(image_path).convert('L') 
img = img.resize((28, 28)) 

plt.imshow(img, cmap='gray')
plt.title("Originalna slika")
plt.axis('off')
plt.show()

img_array = np.array(img) 
img_array = img_array.astype("float32") / 255 
img_array = np.expand_dims(img_array, axis=-1) 
img_array = np.expand_dims(img_array, axis=0)  

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

print("Predviđeni broj:", predicted_class)

# -- Predviđen je broj 2 kao što je i na slici.
# -- Kada promjenim sliku na broj 5, također dobro predviđa. 