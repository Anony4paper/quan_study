import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist, cifar10
import tensorflow_model_optimization as tfmot
import argparse
from tensorflow_addons.optimizers import SGDW
import keras
tf.keras.optimizers.SGDW = SGDW


def representative_dataset():
  for _ in range(100):
    data = np.random.rand(1, 28, 28, 1)
    yield [data.astype(np.float32)]


# def representative_data_gen(data_type):
#     if data_type == 'mnist':
#         (x_train, y_train), (_, _) = mnist.load_data()
#         x_train = x_train.reshape(-1, 28, 28, 1)
#         x_train = x_train.astype('float32') / 255
#     elif data_type == 'imagenet':
#         x_train = np.load("../datasets/tiny-imagenet/seed.npy")
#     elif data_type == 'iwildscam':
#         x_train = np.load("../datasets/tiny-iwildscam/seed.npy")
#     else:
#         max_features = 20000
#         maxlen = 200
#         (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
#             num_words=max_features
#         )
#         x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
#     for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100):
#         # Model has only one input so each data point has one element.
#         yield [input_value]


def representative_data_gen_mnist():
    (x_train, y_train), (_, _) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_train = x_train.astype('float32') / 255
    for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100):
        # Model has only one input so each data point has one element.
        yield [input_value]


def representative_data_gen_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100):
        # Model has only one input so each data point has one element.
        yield [input_value]


def representative_data_gen_imagenet():
    x_train = np.load("../datasets/tiny-imagenet/seed.npy")
    for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100):
        # Model has only one input so each data point has one element.
        yield [input_value]


def representative_data_gen_imagenet_test():
    rng = np.random.default_rng()
    x_train = rng.random(size=(3, 224, 224, 3), dtype=np.float32)
    for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100):
        # Model has only one input so each data point has one element.
        yield [input_value]


def representative_data_gen_iwilds():
    x_train = np.load("../datasets/iwildcam/seed.npy")
    for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100):
        # Model has only one input so each data point has one element.
        yield [input_value]


def representative_data_gen_imdb():
    max_features = 20000
    maxlen = 200
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.imdb.load_data(
        num_words=max_features
    )
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100):
        # Model has only one input so each data point has one element.
        yield [input_value]


def model_covert(h5_path, save_path, data_type, qa_training=False, num_bits=16):
  if qa_training:
    with tfmot.quantization.keras.quantize_scope():
        model = tf.keras.models.load_model(h5_path)
        # model.summary()
  else:
    model = tf.keras.models.load_model(h5_path)
  model.summary()
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  if num_bits == 8:
    if data_type == 'mnist':
        # print("hahaha")
        converter.representative_dataset = representative_data_gen_mnist
    if data_type == 'cifar10':
        converter.representative_dataset = representative_data_gen_cifar10
    elif data_type == 'imagenet':
        converter.representative_dataset = representative_data_gen_imagenet
        converter._experimental_lower_tensor_list_ops = True
    elif data_type == 'iwildscam':
        converter.representative_dataset = representative_data_gen_iwilds
    else:
        converter.representative_dataset = representative_data_gen_imdb
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.allow_custom_ops = True
    converter.experimental_new_quantizer = True
  tflite_quant_model = converter.convert()
  with open(save_path, 'wb') as f:
    f.write(tflite_quant_model)


def main_tflite():
  parser = argparse.ArgumentParser()
  parser.add_argument("--h5_path", "-h5_path",
                      type=str,
                      default="../models/mnist/lenet1.h5"
                      )
  parser.add_argument("--save_path", "-save_path",
                      type=str,
                      default="../models/mnist/lenet1-8bit-quan.lite"
                      )
  parser.add_argument("--data_type", "-data_type",
                      type=str,
                      default="mnist"
                      )
  parser.add_argument("--qa_training", "-qa_training",
                      type=bool,
                      default=False
                      )
  parser.add_argument("--bits", "-bits",
                      type=int,
                      default=8
                      )
  args = parser.parse_args()
  h5_path = args.h5_path
  save_path = args.save_path
  data_type = args.data_type
  qa_training = args.qa_training
  num_bits = args.bits
  model_covert(h5_path, save_path, data_type, qa_training, num_bits)


if __name__ == '__main__':
  main_tflite()

# python keras2tflite.py -h5_path ../models/RQ4/wilds/densenet.h5 -save_path ../models/RQ4/wilds/tflite/densenet_8bits.lite -data_type iwildscam -bits 8
# python keras2tflite.py -h5_path ../models/RQ4/wilds/densenet.h5 -save_path ../models/RQ4/wilds/tflite/densenet_16bits.lite -data_type iwildscam -bits 16
# python keras2tflite.py -h5_path ../models/RQ4/wilds/resnet50.h5 -save_path ../models/RQ4/wilds/tflite/resnet50_8bits.lite -data_type iwildscam -bits 8
# python keras2tflite.py -h5_path ../models/RQ4/wilds/resnet50.h5 -save_path ../models/RQ4/wilds/tflite/resnet50_16bits.lite -data_type iwildscam -bits 16


# python keras2tflite.py -h5_path ../models/RQ4/mnist/lenet5.h5 -save_path ../models/RQ4/mnist/lenet5_16bits.lite -data_type mnist -bits 16
# python keras2tflite.py -h5_path ../models/mnist/lenet5.h5 -save_path ../models/RQ4/mnist/test.lite -data_type mnist -bits 8
# python keras2tflite.py -h5_path ../models/cifar10/nin.h5 -save_path ../models/RQ4/mnist/test.lite -data_type mnist -bits 8
# python keras2tflite.py -h5_path ../models/iwildscam/densenet_adv_2.h5 -save_path ../models/iwildscam/tflite/densenet_adv2_16bits.lite -data_type iwildscam -bits 16
# python keras2tflite.py -h5_path ../models/iwildscam/resnet50_adv_2.h5 -save_path ../models/iwildscam/tflite/resnet50_adv2_16bits.lite -data_type iwildscam -bits 16
