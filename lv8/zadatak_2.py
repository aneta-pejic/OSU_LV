import numpy as np
import tensorflow as tf
import keras
from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_test_s = x_test.astype("float32") / 255

x_test_s = np.expand_dims(x_test_s, -1)

model = keras.models.load_model("mnist_model.h5")

y_pred = model.predict(x_test_s)
y_pred_classes = np.argmax(y_pred, axis=1)  

misclassified_indices = np.where(y_pred_classes != y_test)[0]

plt.figure(figsize=(10, 10))
for i, idx in enumerate(misclassified_indices[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(y_test[idx], y_pred_classes[idx])
    plt.axis('off')
plt.suptitle("Loše klasificirane slike")
plt.show()