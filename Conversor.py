import PruebaPy

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('/home/paupcp/TFM/Codigo/TFM/Zalando.h5') # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
