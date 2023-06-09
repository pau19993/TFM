# TensorFlow y tf.keras
import tensorflow as tf
from tensorflow import keras

# Librerias de ayuda
import numpy as np
import matplotlib.pyplot as plt

#Ospath
import os.path

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape

len(train_labels)
train_labels
test_images.shape
len(test_labels)


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
#plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

"""plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
"""
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

if os.path.isfile('/home/paupcp/TFM/Codigo/TFM/Zalando.h5') is False:
    model.save('/home/paupcp/TFM/Codigo/TFM/Zalando.h5')

saved_model = '/home/paupcp/TFM/Codigo/TFM/Zalando.h5'

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(saved_model)
#converter.inference_input_type = tf.uint8  # or tf.uint8
#converter.inference_output_type = tf.uint8  # or tf.uint8
tflite_model = converter.convert()

# Save the model.
with open('model2.tflite', 'wb') as f:
  f.write(tflite_model)


print('\nTest accuracy:', test_acc)

print("Esto sigue siendo una prueba")
