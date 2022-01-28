from coremltools.models.neural_network import quantization_utils
import coremltools
import tensorflow as tf
from tensorflow_addons.optimizers import SGDW
import tensorflow_model_optimization as tfmot
import argparse
from coremltools.models.neural_network.quantization_utils import AdvancedQuantizedLayerSelector
import keras
tf.keras.optimizers.SGDW = SGDW


selector = AdvancedQuantizedLayerSelector(
    # skip_layer_types=['batchnorm', 'bias', 'depthwiseConv'],
    skip_layer_types=['batchnorm', 'depthwiseConv'],
    minimum_conv_kernel_channels=4,
    minimum_conv_weight_count=4096
)


def model_covert_coreml(h5_path, save_path, bit_num, qa_training=False):
    if qa_training:
        with tfmot.quantization.keras.quantize_scope():
            model = tf.keras.models.load_model(h5_path)
    else:
        model = keras.models.load_model(h5_path)
    coreml_model = coremltools.convert(model)
    coreml_quan = quantization_utils.quantize_weights(coreml_model,
                                                      nbits=bit_num,
                                                      # quantization_mode="linear_symmetric",
                                                      selector=selector
                                                      )
    coreml_model.author = "wellido"
    coreml_model.license = 'MIT'
    coreml_quan.save(save_path)


def main_coreml():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", "-h5_path",
                        type=str
                        )
    parser.add_argument("--save_path", "-save_path",
                        type=str
                        )
    parser.add_argument("--bits", "-bits",
                        type=int,
                        default=8
                        )
    parser.add_argument("--qa_training", "-qa_training",
                        type=bool,
                        default=False
                        )
    args = parser.parse_args()
    h5_path = args.h5_path
    save_path = args.save_path
    bit_num = args.bits
    qa_training = args.qa_training
    model_covert_coreml(h5_path, save_path, bit_num, qa_training)


if __name__ == '__main__':
    main_coreml()


# python keras2coreml.py -h5_path ../models/RQ4/wilds/densenet.h5 -save_path ../models/RQ4/wilds/coreml/densenet_8bits.mlmodel -bits 8
# python keras2coreml.py -h5_path ../models/RQ4/wilds/densenet.h5 -save_path ../models/RQ4/wilds/coreml/densenet_16bits.mlmodel -bits 16
# python keras2coreml.py -h5_path ../models/RQ4/wilds/resnet50.h5 -save_path ../models/RQ4/wilds/coreml/resnet50_8bits.mlmodel -bits 8
# python keras2coreml.py -h5_path ../models/RQ4/wilds/resnet50.h5 -save_path ../models/RQ4/wilds/coreml/resnet50_16bits.mlmodel -bits 16


# python keras2coreml.py -h5_path ../models/RQ4/cifar10/lenet5.h5 -save_path ../models/RQ4/mnist/lenet5_16bits.mlmodel -bits 16
# python keras2coreml.py -h5_path ../models/RQ4/cifar10/lenet5.h5 -save_path ../models/RQ4/mnist/lenet5_16bits.mlmodel -bits 16

# python keras2coreml.py -h5_path ../models/imagenet/resnet101_adv.h5 -save_path ../models/imagenet/coreml/resnet101_adv-8bit-quan.mlmodel -bits 8
# python keras2coreml.py -h5_path ../models/imdb/gru_5000.h5 -save_path ../models/imdb/coreml/gru-16bits-quan.mlmodel -bits 16
# python keras2coreml.py -h5_path ../models/iwildscam/densenet_adv_2.h5 -save_path ../models/iwildscam/coreml/densenet_adv2_8bits.mlmodel -bits 8
# python keras2coreml.py -h5_path ../models/iwildscam/resnet50_adv_2.h5 -save_path ../models/iwildscam/coreml/resnet50_adv2_8bits.mlmodel -bits 8
