'''
LeNet-5
'''

# usage: python MNISTModel3.py - train the model
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.regularizers import l2
from keras.layers import Conv2D, Dense, Input, add, Activation, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization
from keras import optimizers, regularizers
from keras.models import Sequential, Model, load_model
from tensorflow.keras.initializers import RandomNormal

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


def NIN_qa(input_shape, num_classes):
    img = Input(shape=input_shape)
    weight_decay = 1e-6
    def NiNBlock(kernel, mlps, strides):
        def inner(x):
            l = Conv2D(mlps[0], kernel, padding='same', strides=strides, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=RandomNormal(stddev = 0.01))(x)
            l = tfmot.quantization.keras.quantize_annotate_layer(BatchNormalization(),
                                                                 quantize_config=quantize_config)(l)
            l = Activation('relu')(l)
            for size in mlps[1:]:
                l = Conv2D(size, 1, padding='same', strides=[1,1], kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=RandomNormal(stddev = 0.05))(l)
                l = tfmot.quantization.keras.quantize_annotate_layer(BatchNormalization(),
                                                                     quantize_config=quantize_config)(l)
                l = Activation('relu')(l)
            return l
        return inner
    l1 = NiNBlock(5, [192, 160, 96], [1,1])(img)
    l1 = tfmot.quantization.keras.quantize_annotate_layer(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding = 'same'),
                                                          quantize_config=quantize_config)(l1)
    l1 = Dropout(0.5)(l1)

    l2 = NiNBlock(5, [192, 192, 192], [1,1])(l1)
    l2 = tfmot.quantization.keras.quantize_annotate_layer(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding = 'same'),
                                                          quantize_config=quantize_config)(l2)
    l2 = Dropout(0.5)(l2)

    l3 = NiNBlock(3, [192, 192, 10], [1,1])(l2)

    l4 = GlobalAveragePooling2D()(l3)
    l4 = Activation('softmax')(l4)

    model = Model(inputs=img, outputs=l4, name='NIN-none')
    return model


def NIN(input_shape, num_classes):
    img = Input(shape=input_shape)
    weight_decay = 1e-6
    def NiNBlock(kernel, mlps, strides):
        def inner(x):
            l = Conv2D(mlps[0], kernel, padding='same', strides=strides, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=RandomNormal(stddev = 0.01))(x)
            l = BatchNormalization()(l)
            l = Activation('relu')(l)
            for size in mlps[1:]:
                l = Conv2D(size, 1, padding='same', strides=[1,1], kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=RandomNormal(stddev = 0.05))(l)
                l = BatchNormalization()(l)
                l = Activation('relu')(l)
            return l
        return inner
    l1 = NiNBlock(5, [192, 160, 96], [1,1])(img)
    l1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding = 'same')(l1)
    l1 = Dropout(0.5)(l1)

    l2 = NiNBlock(5, [192, 192, 192], [1,1])(l1)
    l2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding = 'same')(l2)
    l2 = Dropout(0.5)(l2)

    l3 = NiNBlock(3, [192, 192, 10], [1,1])(l2)

    l4 = GlobalAveragePooling2D()(l3)
    l4 = Activation('softmax')(l4)

    model = Model(inputs=img, outputs=l4, name='NIN-none')
    return model


def train_nin():
    input_shape = (32, 32, 3)
    model = NIN(input_shape, 10)
    model.summary()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    model_filename = "../models/cifar10/nin.h5"
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
    # model.save("../../new_models/RQ1/NiN/NiN_" + str(num) + ".h5")
    # print(history.history['val_accuracy'][-1])


if __name__ == '__main__':
    train_nin()
