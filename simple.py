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
input_shape = (31, 31, 1)  # Actualizar a una dimensión de tamaño 4

# Clases de ropa correspondientes a las salidas del modelo
class_names = ['Camiseta', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo', 'Sandalia', 'Camisa', 'Zapatilla deportiva', 'Bolso', 'Bota']

# Seleccionar una imagen de la carpeta
image_path = 'images.jpg'
image = Image.open(image_path).resize(input_shape[:2])  # Redimensionar solo las dimensiones espaciales

# Preprocesamiento de la imagen
image_array = np.array(image) / 255.0

# Asegurarse de que las dimensiones de la imagen sean correctas
if image_array.shape != input_shape[:2]:
    image_array = np.resize(image_array, input_shape[:2])

# Agregar una dimensión adicional para representar el batch de imágenes y el canal
input_data = np.expand_dims(image_array, axis=0)
input_data = np.expand_dims(input_data, axis=-1)

# Convertir el tipo de datos a FLOAT32
input_data = input_data.astype(np.float32)

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
