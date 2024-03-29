from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator

from keras.regularizers import l2
from keras.layers import Conv2D, Dense, Input, add, Activation, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization
from keras import optimizers, regularizers
from keras.models import Sequential, Model, load_model

from keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten,MaxPool2D, Dropout

from keras.models import Model
from keras.datasets import cifar10
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.quantization.keras import quantize_config


class NoOpQuantizeConfig(quantize_config.QuantizeConfig):
    """QuantizeConfig which does not quantize any part of the layer."""
    def get_weights_and_quantizers(self, layer):
        return []
    def get_activations_and_quantizers(self, layer):
        return []
    def set_quantize_weights(self, layer, quantize_weights):
        pass
    def set_quantize_activations(self, layer, quantize_activations):
        pass
    def get_output_quantizers(self, layer):
        return []
    def get_config(self):
        return {}


quantize_config = NoOpQuantizeConfig()


def resnet20(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)
    weight_decay = 1e-6
    stack_n = 3
    def residual_block(intput, out_channel, increase=False):
        if increase:
            stride = (2, 2)
        else:
            stride = (1, 1)

        pre_bn = BatchNormalization()(intput)
        pre_relu = Activation('relu')(pre_bn)

        conv_1 = Conv2D(out_channel, kernel_size=(3, 3), strides=stride, padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(pre_relu)
        bn_1 = BatchNormalization()(conv_1)
        relu1 = Activation('relu')(bn_1)
        conv_2 = Conv2D(out_channel, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(relu1)
        conv_2 = Dropout(0.5)(conv_2)
        if increase:
            projection = Conv2D(out_channel,
                                kernel_size=(1, 1),
                                strides=(2, 2),
                                padding='same',
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(intput)
            block = add([conv_2, projection])
        else:
            block = add([intput, conv_2])
        return block

        # build model
        # total layers = stack_n * 3 * 2 + 2
        # stack_n = 5 by default, total layers = 32, which is resnet32
        # input: 32x32x3 output: 32x32x16

    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay))(input_tensor)
    # input: 32x32x16 output: 32x32x16
    for _ in range(stack_n):
        x = residual_block(x, 16, False)

    # input: 32x32x16 output: 16x16x32
    x = residual_block(x, 32, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 32, False)

    # input: 16x16x32 output: 8x8x64
    x = residual_block(x, 64, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 64, False)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(num_classes, name='before_softmax',
              kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('softmax')(x)
    model = Model(input_tensor, x, name='res20-none')
    return model


def resnet20_qa(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)
    weight_decay = 1e-6
    stack_n = 3
    def residual_block(intput, out_channel, increase=False):
        if increase:
            stride = (2, 2)
        else:
            stride = (1, 1)

        pre_bn = tfmot.quantization.keras.quantize_annotate_layer(BatchNormalization(),
                                                                  quantize_config=quantize_config)(intput)
        pre_relu = Activation('relu')(pre_bn)

        conv_1 = Conv2D(out_channel, kernel_size=(3, 3), strides=stride, padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(pre_relu)
        bn_1 = tfmot.quantization.keras.quantize_annotate_layer(BatchNormalization(),
                                                                quantize_config=quantize_config)(conv_1)
        relu1 = Activation('relu')(bn_1)
        conv_2 = Conv2D(out_channel, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(relu1)
        conv_2 = Dropout(0.5)(conv_2)
        if increase:
            projection = Conv2D(out_channel,
                                kernel_size=(1, 1),
                                strides=(2, 2),
                                padding='same',
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(intput)
            block = add([conv_2, projection])
        else:
            block = add([intput, conv_2])
        return block

        # build model
        # total layers = stack_n * 3 * 2 + 2
        # stack_n = 5 by default, total layers = 32, which is resnet32
        # input: 32x32x3 output: 32x32x16

    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay))(input_tensor)
    # input: 32x32x16 output: 32x32x16
    for _ in range(stack_n):
        x = residual_block(x, 16, False)

    # input: 32x32x16 output: 16x16x32
    x = residual_block(x, 32, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 32, False)

    # input: 16x16x32 output: 8x8x64
    x = residual_block(x, 64, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 64, False)

    x = tfmot.quantization.keras.quantize_annotate_layer(BatchNormalization(),
                                                         quantize_config=quantize_config)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(num_classes, name='before_softmax',
              kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('softmax')(x)
    model = Model(input_tensor, x, name='res20-none')
    return model


def train_resnet():
    input_shape = (32, 32, 3)
    model = resnet20(input_shape, 10)
    model.summary()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    model_filename = "../models/cifar10/resnet20.h5"

    checkpoint = ModelCheckpoint(model_filename,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 period=1)
    # cbks = [checkpoint]
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=200,
                        validation_data=(x_test, y_test),
                        verbose=1,
                        callbacks=[checkpoint]
                        )


if __name__ == '__main__':

    train_resnet()

