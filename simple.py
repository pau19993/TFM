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
foto = np.array(image)[:-1]
classify.set_input(interpreter, foto)

"""
interpreter.set_tensor(input_index, features.astype(np.float32))
interpreter.invoke()
prediction = interpreter.get_tensor(output_index)
"""

start = time.perf_counter()
interpreter.invoke()
inference_time = time.perf_counter() - start
classes = classify.get_output(interpreter, 1, 0)
print('%.1fms' % (inference_time * 1000))

print('-------RESULTS--------')
for klass in classes:
    print('%s: %.5f' % (labels.get(klass.id, klass.id), klass.score))
