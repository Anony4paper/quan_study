import tempfile
import os

import tensorflow as tf
import sys
from tensorflow import keras
import tensorflow_addons as tfa
import argparse
import tensorflow_model_optimization as tfmot
import numpy as np
from tensorflow_model_optimization.python.core.quantization.keras import quantize_config
# from tensorflow_model_optimization.quantization.keras import QuantizeConfig

sys.path.append('../model_prepare')
from imagenet_models import *
from IMDB import *
from iWildsCam import *
from LeNet import *
from Newsgroups import *
from data_prepare_tiny import *
from ResNet101_original import ResNet101_QA_model
from DenseNet121_original import DenseNet_QA_model, DenseNet_QA_model_iwilds
from ResNet50_original import ResNet50_QA_model
from ResNet20 import *
from NiN import *


def schedule(epoch_idx):
    if (epoch_idx + 1) < 10:
        return 1e-03
    elif (epoch_idx + 1) < 20:
        return 1e-04  # lr_decay_ratio = 0.2
    elif (epoch_idx + 1) < 30:
        return 1e-05
    return 1e-05


###   Clone NoOpQuantizeConfig locally   ###
class NoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
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


def annotate(layer):
    if layer._name.startswith('tf_op_layer_ResizeBilinear'):
        return layer   # pass thru; don't quantize tf.image.resize()
    # quantize everything else
    return tfmot.quantization.keras.quantize_annotate_layer(layer)


def qa_training(data_type, model_type, save_path):
    if data_type == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        y_train = tf.keras.utils.to_categorical(y_train, 10)

        batch_size = 256
        epochs = 20
        if model_type == 'lenet1':
            model = LeNet1_model()
        else:
            model = LeNet5_model()
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        his = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        shuffle=True,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        verbose=1)
        quantize_model = tfmot.quantization.keras.quantize_model
        q_aware_model = quantize_model(model)
        q_aware_model.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])

        q_aware_model.summary()
        q_aware_model.fit(x_train,
                          y_train,
                          batch_size=batch_size,
                          shuffle=True,
                          epochs=epochs,
                          validation_data=(x_test, y_test)
                          )
        q_aware_model.save(save_path)
    elif data_type == 'cifar10':
        input_shape = (32, 32, 3)
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        batch_size = 128
        epochs = 200
        if model_type == 'resnet20':
            model = resnet20_qa(input_shape, 10)
        else:
            model = NIN_qa(input_shape, 10)

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                      metrics=['accuracy'])
        model.fit(x_train,
                  y_train,
                  batch_size=batch_size,
                  epochs=100,
                  validation_data=(x_test, y_test),
                  verbose=1
                  )
        with tf.keras.utils.custom_object_scope({'NoOpQuantizeConfig': NoOpQuantizeConfig}):
            quantize_model = tfmot.quantization.keras.quantize_model
            q_aware_model = quantize_model(model)
        q_aware_model.compile(loss='categorical_crossentropy',
                              optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                              metrics=['accuracy'])
        checkpoint = ModelCheckpoint(save_path,
                                     monitor='val_accuracy',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max',
                                     period=1)
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  verbose=1,
                  callbacks=[checkpoint]
                  )
    elif data_type == 'imagenet':
        training_folder = '../datasets/tiny-imagenet/tiny-imagenet-200/train/'
        val_folder = '../datasets/tiny-imagenet/tiny-imagenet-200/val/'
        train_generator, val_generator = Tiny_generator(training_folder, val_folder)
        config = imagenet_config()
        config.set_defaults()
        checkPoint = tf.keras.callbacks.ModelCheckpoint(save_path, monitor="val_accuracy",
                                                        save_best_only=True, verbose=0)
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(schedule=schedule)
        cbs = [lr_schedule, checkPoint]
        if model_type == 'resnet101':
            # model = ResNet101_QA_model()
            with tfmot.quantization.keras.quantize_scope():
                q_aware_model = tf.keras.models.load_model("../models/imagenet/resnet101_qa.h5")
        else:
            # model = DenseNet_QA_model()
            with tfmot.quantization.keras.quantize_scope():
                q_aware_model = tf.keras.models.load_model("../models/imagenet/densenet_qa.h5")
        q_aware_model.summary()
        # model.compile(loss='categorical_crossentropy',
        #               optimizer=tfa.optimizers.SGDW(lr=1e-03, momentum=0.9, weight_decay=1e-4),
        #               # optimizer=optimizers.Adam(lr=1e-03),
        #               metrics=['accuracy'])
        # model.fit_generator(generator=train_generator,
        #                     epochs=config.epochs,
        #                     steps_per_epoch=100000 // config.batch_size,
        #                     validation_data=val_generator,
        #                     validation_steps=10000 // config.batch_size,
        #                     use_multiprocessing=True,
        #                     callbacks=cbs)
        # # model.save(save_path)
        # with tfmot.quantization.keras.quantize_scope():
        #     model = tf.keras.models.load_model(save_path)

        # with tf.keras.utils.custom_object_scope({'NoOpQuantizeConfig': NoOpQuantizeConfig}):
        # quantize_model = tfmot.quantization.keras.quantize_model
        # q_aware_model = quantize_model(model)
        q_aware_model.compile(loss='categorical_crossentropy',
                              optimizer=tfa.optimizers.SGDW(lr=1e-03, momentum=0.9, weight_decay=1e-4),
                              # optimizer=optimizers.Adam(lr=1e-03),
                              metrics=['accuracy'])
        q_aware_model.fit_generator(generator=train_generator,
                                    epochs=config.epochs,
                                    steps_per_epoch=100000 // config.batch_size,
                                    validation_data=val_generator,
                                    validation_steps=10000 // config.batch_size,
                                    use_multiprocessing=True,
                                    callbacks=cbs)
    elif data_type == 'imdb':
        max_example_len = 200
        batch_size = 32
        embedding_dims = 50
        vocab_size = 19999
        num_epochs = 10
        df = pd.read_csv("../datasets/IMDB/IMDB_Dataset.csv")
        df['review_cleaned'] = df['review'].apply(lambda x: preprocessing_text(x))
        x_train = df['review_cleaned'][:5000]
        tokenizer = Tokenizer(num_words=20000)
        tokenizer.fit_on_texts(x_train)
        x_train = tokenizer.texts_to_sequences(x_train)
        max_example_len = 200
        x_train = pad_sequences(x_train, maxlen=max_example_len)
        y = df['sentiment'].map({'negative': 0, 'positive': 1}).values
        y_train = y[:5000]
        y_train = to_categorical(y_train, 2)
        x_test = df['review_cleaned'][25000:30000]
        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen=max_example_len)
        y_test = y[25000:30000]
        y_test = to_categorical(y_test, 2)
        if model_type == 'lstm':
            # model = tf.keras.models.load_model("../models/imdb/IMDB_lstm.h5")
            model = IMDB_LSTM_QA()
        else:
            # model = tf.keras.models.load_model("../models/imdb/IMDB_gru.h5")
            model = IMDB_GRU_QA()

        model.compile('adam',
                      # "categorical_crossentropy",
                      "binary_crossentropy",
                      metrics=["accuracy"])
        model.fit(x_train,
                  y_train,
                  epochs=5,
                  batch_size=batch_size,
                  validation_data=(x_test, y_test)
                  )
        with tf.keras.utils.custom_object_scope({'NoOpQuantizeConfig': NoOpQuantizeConfig}):
            # quantize_model = tfmot.quantization.keras.quantize_model
            # q_aware_model = quantize_model(model)
            q_aware_model = tfmot.quantization.keras.quantize_apply(model)
        # quantize_model = tfmot.quantization.keras.quantize_model
        # q_aware_model = quantize_model(model)
        # q_aware_model = tfmot.quantization.keras.quantize_apply(model)

        q_aware_model.compile('adam',
                              # "categorical_crossentropy",
                              "binary_crossentropy",
                              metrics=["accuracy"])

        checkPoint = tf.keras.callbacks.ModelCheckpoint(save_path, monitor="val_accuracy",
                                                        save_best_only=True, verbose=0)
        q_aware_model.fit(x_train,
                          y_train,
                          epochs=num_epochs,
                          batch_size=batch_size,
                          validation_data=(x_test, y_test),
                          callbacks=[checkPoint])

    elif data_type == 'iwildscam':
        dataset = get_dataset(dataset='iwildcam', download=True)
        train_data = dataset.get_subset('train',
                                        transform=transforms.Compose([transforms.Resize((448, 448)),
                                                                      transforms.ToTensor()]))
        val_data = dataset.get_subset('val',
                                      transform=transforms.Compose([transforms.Resize((448, 448)),
                                                                    transforms.ToTensor()]))

        train_loader = get_train_loader('standard', train_data, batch_size=16)
        val_loader = get_train_loader('standard', val_data, batch_size=16)
        dataloader = DataGenerator(train_loader, 182)
        val_dataloader = DataGenerator(val_loader, 182)
        if model_type == 'resnet50':
            # model = ResNet50_QA_model()
            with tfmot.quantization.keras.quantize_scope():
                q_aware_model = tf.keras.models.load_model("../models/iwildscam/resnet50_qa.h5")
        else:
            # model = DenseNet_QA_model_iwilds()
            with tfmot.quantization.keras.quantize_scope():
                q_aware_model = tf.keras.models.load_model("../models/iwildscam/densenet_qa.h5")

        checkPoint = tf.keras.callbacks.ModelCheckpoint(save_path, monitor="val_accuracy",
                                                        save_best_only=True, verbose=1)
        steps_per_epoch = 129809 // 16
        valid_per_step = 7314 // 16
        # model.compile(loss='categorical_crossentropy',
        #               optimizer=keras.optimizers.Adam(lr=3e-5),
        #               metrics=['accuracy'])
        # model.fit_generator(dataloader,
        #                     steps_per_epoch=steps_per_epoch,
        #                     validation_data=val_dataloader,
        #                     validation_steps=valid_per_step,
        #                     epochs=10,
        #                     callbacks=[checkPoint])
        #
        # with tfmot.quantization.keras.quantize_scope():
        #     model = tf.keras.models.load_model(save_path)
        # with tf.keras.utils.custom_object_scope({'NoOpQuantizeConfig': NoOpQuantizeConfig}):
        #     quantize_model = tfmot.quantization.keras.quantize_model
        #     q_aware_model = quantize_model(model)
        q_aware_model.compile(loss='categorical_crossentropy',
                              optimizer=keras.optimizers.Adam(lr=3e-5),
                              metrics=['accuracy'])

        q_aware_model.fit_generator(dataloader,
                                    steps_per_epoch=steps_per_epoch,
                                    validation_data=val_dataloader,
                                    validation_steps=valid_per_step,
                                    epochs=10,
                                    callbacks=[checkPoint])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-data",
                        type=str,
                        default='imdb'
                        )
    parser.add_argument("--model", "-model",
                        type=str,
                        default='gru'
                        )
    parser.add_argument("--save", "-save",
                        type=str,
                        default='../models/imdb/gru_qa.h5'
                        )
    args = parser.parse_args()
    data_type = args.data
    model_type = args.model
    save_path = args.save
    qa_training(data_type, model_type, save_path)

# python Q_aware_training.py
