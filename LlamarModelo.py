import tflite_runtime.interpreter as tflite
import model.tflite

model =  keras.model.load_model("model.tflite")

