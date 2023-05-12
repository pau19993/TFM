""" how to call this model= python3 LlamarModelo.py \
  --model model.tflite  \
  --labels labels.txt \
  --input images.jpeg
"""


import argparse
import re
import threading
import time

import numpy as np
from PIL import Image

from pycoral.adapters import classify
from pycoral.adapters import common
import pycoral.pipeline.pipelined_model_runner as pipeline
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import list_edge_tpus
from pycoral.utils.edgetpu import make_interpreter


def _get_devices(num_devices):
  """Returns list of device names in usb:N or pci:N format.

  This function prefers returning PCI Edge TPU first.

  Args:
    num_devices: int, number of devices expected

  Returns:
    list of devices in pci:N and/or usb:N format

  Raises:
    RuntimeError: if not enough devices are available
  """
  edge_tpus = list_edge_tpus()

  if len(edge_tpus) < num_devices:
    raise RuntimeError(
        'Not enough Edge TPUs detected, expected %d, detected %d.' %
        (num_devices, len(edge_tpus)))

  num_pci_devices = sum(1 for device in edge_tpus if device['type'] == 'pci')

  return ['pci:%d' % i for i in range(min(num_devices, num_pci_devices))] + [
      'usb:%d' % i for i in range(max(0, num_devices - num_pci_devices))
  ]


def _make_runner(model_paths, devices):
  """Constructs PipelinedModelRunner given model paths and devices."""
  print('Using devices: ', devices)
  print('Using models: ', model_paths)

  if len(model_paths) != len(devices):
    raise ValueError('# of devices and # of model_paths should match')

  interpreters = [make_interpreter(m, d) for m, d in zip(model_paths, devices)]
  for interpreter in interpreters:
    interpreter.allocate_tensors()
  return pipeline.PipelinedModelRunner(interpreters)


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-m',
      '--models',
      required=True,
      help=('File path template of .tflite model segments, e.g.,'
            'inception_v3_299_quant_segment_%d_of_2_edgetpu.tflite'))
  parser.add_argument(
      '-i', '--input', required=True, help='Image to be classified.')
  parser.add_argument('-l', '--labels', help='File path of labels file.')
  parser.add_argument(
      '-k',
      '--top_k',
      type=int,
      default=1,
      help='Max number of classification results')
  parser.add_argument(
      '-t',
      '--threshold',
      type=float,
      default=0.0,
      help='Classification score threshold')
  parser.add_argument(
      '-c',
      '--count',
      type=int,
      default=5,
      help='Number of times to run inference')
  args = parser.parse_args()
  labels = read_label_file(args.labels) if args.labels else {}

  result = re.search(r'^.*_segment_%d_of_(?P<num_segments>[0-9]+)_.*.tflite',
                     args.models)
  if not result:
    raise ValueError(
        '--models should follow *_segment%d_of_[num_segments]_*.tflite pattern')
  num_segments = int(result.group('num_segments'))
  model_paths = [args.models % i for i in range(num_segments)]
  devices = _get_devices(num_segments)
  runner = _make_runner(model_paths, devices)

  size = common.input_size(runner.interpreters()[0])
  name = common.input_details(runner.interpreters()[0], 'name')
  image = np.array(
      Image.open(args.input).convert('RGB').resize(size, Image.ANTIALIAS))

  def producer():
    for _ in range(args.count):
      runner.push({name: image})
    runner.push({})

  def consumer():
    output_details = runner.interpreters()[-1].get_output_details()[0]
    scale, zero_point = output_details['quantization']
    while True:
      result = runner.pop()
      if not result:
        break
      values, = result.values()
      scores = scale * (values[0].astype(np.int64) - zero_point)
      classes = classify.get_classes_from_scores(scores, args.top_k,
                                                 args.threshold)
    print('-------RESULTS--------')
    for klass in classes:
      print('%s: %.5f' % (labels.get(klass.id, klass.id), klass.score))

  start = time.perf_counter()
  producer_thread = threading.Thread(target=producer)
  consumer_thread = threading.Thread(target=consumer)
  producer_thread.start()
  consumer_thread.start()
  producer_thread.join()
  consumer_thread.join()
  average_time_ms = (time.perf_counter() - start) / args.count * 1000
  print('Average inference time (over %d iterations): %.1fms' %
        (args.count, average_time_ms))


if __name__ == '__main__':
  main()

"""import tflite_runtime.interpreter as tflite
import argparse
import time

import numpy as np
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-m', '--model', required=True, help='File path of .tflite file.')
  parser.add_argument(
      '-i', '--input', required=True, help='Image to be classified.')
  parser.add_argument(
      '-l', '--labels', help='File path of labels file.')
  parser.add_argument(
      '-k', '--top_k', type=int, default=1,
      help='Max number of classification results')
  parser.add_argument(
      '-t', '--threshold', type=float, default=0.0,
      help='Classification score threshold')
  parser.add_argument(
      '-c', '--count', type=int, default=5,
      help='Number of times to run inference')
  parser.add_argument(
      '-a', '--input_mean', type=float, default=128.0,
      help='Mean value for input normalization')
  parser.add_argument(
      '-s', '--input_std', type=float, default=128.0,
      help='STD value for input normalization')
  args = parser.parse_args()

  labels = read_label_file(args.labels) if args.labels else {}

  interpreter = make_interpreter(*args.model.split('@'))
  interpreter.allocate_tensors()

  # Model must be uint8 quantized
  if common.input_details(interpreter, 'dtype') != np.uint8:
    raise ValueError('Only support uint8 input type.')

  size = common.input_size(interpreter)
  image = Image.open(args.input).convert('RGB').resize(size, Image.ANTIALIAS)

  # Image data must go through two transforms before running inference:
  # 1. normalization: f = (input - mean) / std
  # 2. quantization: q = f / scale + zero_point
  # The following code combines the two steps as such:
  # q = (input - mean) / (std * scale) + zero_point
  # However, if std * scale equals 1, and mean - zero_point equals 0, the input
  # does not need any preprocessing (but in practice, even if the results are
  # very close to 1 and 0, it is probably okay to skip preprocessing for better
  # efficiency; we use 1e-5 below instead of absolute zero).
  params = common.input_details(interpreter, 'quantization_parameters')
  scale = params['scales']
  zero_point = params['zero_points']
  mean = args.input_mean
  std = args.input_std
  if abs(scale * std - 1) < 1e-5 and abs(mean - zero_point) < 1e-5:
    # Input data does not require preprocessing.
    common.set_input(interpreter, image)
  else:
    # Input data requires preprocessing
    normalized_input = (np.asarray(image) - mean) / (std * scale) + zero_point
    np.clip(normalized_input, 0, 255, out=normalized_input)
    common.set_input(interpreter, normalized_input.astype(np.uint8))

  # Run inference
  print('----INFERENCE TIME----')
  print('Note: The first inference on Edge TPU is slow because it includes',
        'loading the model into Edge TPU memory.')
  for _ in range(args.count):
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    classes = classify.get_classes(interpreter, args.top_k, args.threshold)
    print('%.1fms' % (inference_time * 1000))

  print('-------RESULTS--------')
  for c in classes:
    print('%s: %.5f' % (labels.get(c.id, c.id), c.score))


if __name__ == '__main__':
  main()
  """


#model =  keras.model.load_model("model.tflite")

