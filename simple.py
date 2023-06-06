
"""
import numpy as np
import classify
import tflite_runtime.interpreter as tflite
import time

from PIL import Image


interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()


input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

size = classify.input_size(interpreter)
image = Image.open('images.jpg').convert('RGB').resize(size, Image.ANTIALIAS)
classify.set_input(interpreter, image)

"""

"""
 ESTO NO SE PARA QUE ES!!!!!!!!!!!!!!!
interpreter.set_tensor(input_index, features.astype(np.float32))
interpreter.invoke()
prediction = interpreter.get_tensor(output_index)
"""

"""

start = time.perf_counter()
interpreter.invoke()
inference_time = time.perf_counter() - start
classes = classify.get_output(interpreter, 1, 0)
print('%.1fms' % (inference_time * 1000))

print('-------RESULTS--------')
for klass in classes:
    print('%s: %.5f' % (labels.get(klass.id, klass.id), klass.score))


"""

import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

# Cargar el modelo TensorFlow Lite (.tflite)
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Obtener los detalles de entrada y salida del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Tamaño de entrada esperado por el modelo
input_shape = (31, 31)

# Clases de ropa correspondientes a las salidas del modelo
class_names = ['Camiseta', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo', 'Sandalia', 'Camisa', 'Zapatilla deportiva', 'Bolso', 'Bota']

# Seleccionar una imagen de la carpeta
image_path = 'images.jpg'
image = Image.open(image_path).resize(input_shape)

# Preprocesamiento de la imagen
image_array = np.array(image) / 255.0
#input_data = np.expand_dims(image_array, axis=0)

# Asegurarse de que las dimensiones de la imagen sean correctas
if image_array.shape != input_shape:
    image_array = np.resize(image_array, input_shape)

# Agregar una dimensión adicional para representar el batch de imágenes
input_data = np.expand_dims(image_array, axis=0)

# Convertir el tipo de datos a FLOAT32
input_data = input_data.astype(np.float32)

# Asegurarse de que las dimensiones del tensor de entrada sean correctas
input_data = np.reshape(input_data, (1,) + input_shape + (1,))

# Establecer los datos de entrada del modelo
interpreter.set_tensor(input_details[0]['index'], input_data)

# Realizar la inferencia
interpreter.invoke()

# Obtener los resultados de la clasificación
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class_index = np.argmax(output_data)
predicted_class = class_names[predicted_class_index]

# Mostrar el resultado en pantalla
print(f'La prenda de ropa en la imagen es: {predicted_class}')
