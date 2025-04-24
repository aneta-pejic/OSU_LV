import numpy as np
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# TODO: prikazi nekoliko slika iz train skupa
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.axis('off')
plt.suptitle('Slike iz train skupa')
plt.show()

# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)


# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu
model = keras.Sequential([
    keras.layers.Input(shape=input_shape),
    keras.layers.Flatten(input_shape=input_shape),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.summary()



# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# TODO: provedi ucenje mreze
batch_size = 32
epochs = 10
history = model.fit(x_train_s, 
                    y_train_s, 
                    batch_size = batch_size, 
                    epochs = epochs, 
                    validation_split=0.1)
                    
predictions = model.predict(x_test_s)
score = model.evaluate(x_test_s, y_test_s, verbose = 0)


# TODO: Prikazi test accuracy i matricu zabune

score = model.evaluate(x_test_s, y_test_s, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

y_pred = model.predict(x_test_s)
y_pred_classes = np.argmax(y_pred, axis=1) 
y_true = np.argmax(y_test_s, axis=1)  

cm = confusion_matrix(y_true, y_pred_classes)
print("Matrica zabune:")
print(cm)


# TODO: spremi model

model.save("mnist_model.h5")



