import numpy as np
import tensorflow as tf
# import keras
from mixup_generator import *
import argparse
import tensorflow_addons as tfa
from textaugment import MIXUP
import pandas as pd
import sys
from tensorflow.keras.datasets import cifar10
sys.path.append('../model_prepare')
from imagenet_models import *
from IMDB import *
from iWildsCam import *
from LeNet import *
from Newsgroups import *
from data_prepare_tiny import *
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


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def mix_up(ds_one, ds_two, alpha=0.2):
    # Unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = tf.shape(images_one)[0]

    # Sample lambda and reshape it to do the mixup
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    images = images_one * x_l + images_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)
    return (images, labels)


def mix_up_training(data_type, model_type, save_path):
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
        AUTO = tf.data.experimental.AUTOTUNE

        train_ds_one = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .shuffle(batch_size * 100)
                .batch(batch_size)
        )
        train_ds_two = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .shuffle(batch_size * 100)
                .batch(batch_size)
        )
        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

        # First create the new dataset using our `mix_up` utility
        train_ds_mu = train_ds.map(
            lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=0.2), num_parallel_calls=AUTO
        )
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(train_ds_mu, epochs=epochs)
        _, test_acc = model.evaluate(test_ds)
        print("Test accuracy: {:.2f}%".format(test_acc * 100))
        model.save(save_path)
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
            model = resnet20(input_shape, 10)
        else:
            model = NIN(input_shape, 10)

        AUTO = tf.data.experimental.AUTOTUNE

        train_ds_one = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .shuffle(batch_size * 100)
                .batch(batch_size)
        )
        train_ds_two = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .shuffle(batch_size * 100)
                .batch(batch_size)
        )
        train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

        # First create the new dataset using our `mix_up` utility
        train_ds_mu = train_ds.map(
            lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=0.2), num_parallel_calls=AUTO
        )
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                      metrics=['accuracy'])
        checkpoint = ModelCheckpoint(save_path,
                                     monitor='val_accuracy',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max',
                                     period=1)
        history = model.fit(train_ds_mu,
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            verbose=1,
                            callbacks=[checkpoint]
                            )
    elif data_type == 'imagenet':
        training_folder = '../datasets/tiny-imagenet/tiny-imagenet-200/train/'
        val_folder = '../datasets/tiny-imagenet/tiny-imagenet-200/val/'
        config = imagenet_config()
        config.set_defaults()
        data_generator = CustomImageDataGen(
            horizontal_flip=True,
        )
        train_generator = MixupImageDataGenerator(generator=data_generator,
                                                  directory=training_folder,
                                                  batch_size=config.batch_size,
                                                  img_height=224,
                                                  img_width=224)
        _, val_generator = Tiny_generator(training_folder, val_folder)

        checkPoint = tf.keras.callbacks.ModelCheckpoint(save_path, monitor="val_accuracy",
                                                        save_best_only=True, verbose=0)
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(schedule=schedule)
        cbs = [lr_schedule, checkPoint]
        if model_type == 'resnet50':
            model = ResNet50_model()
        elif model_type == 'resnet101':
            model = ResNet101_model()
        elif model_type == 'efficientnet':
            model = Efficientnet_model()
        else:
            model = DenseNet_model()
        model.summary()
        # loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(loss='categorical_crossentropy',
                      optimizer=tfa.optimizers.SGDW(lr=1e-03, momentum=0.9, weight_decay=1e-4),
                      # optimizer=optimizers.Adam(lr=1e-03),
                      metrics=['accuracy'])
        model.fit_generator(generator=train_generator,
                            epochs=30,
                            steps_per_epoch=train_generator.get_steps_per_epoch(),
                            validation_data=val_generator,
                            validation_steps=10000 // config.batch_size,
                            # use_multiprocessing=True,
                            callbacks=cbs)
    elif data_type == 'imdb':
        max_example_len = 200
        batch_size = 32
        embedding_dims = 50
        vocab_size = 19999
        num_epochs = 20
        df = pd.read_csv("../datasets/IMDB/IMDB_Dataset.csv")
        df['review_cleaned'] = df['review'].apply(lambda x: preprocessing_text(x))
        # x_train = df['review_cleaned'][:5000]
        x_train = df['review_cleaned'][:25000]
        tokenizer = Tokenizer(num_words=20000)
        tokenizer.fit_on_texts(x_train)
        x_train = tokenizer.texts_to_sequences(x_train)
        max_example_len = 200
        x_train = pad_sequences(x_train, maxlen=max_example_len)
        y = df['sentiment'].map({'negative': 0, 'positive': 1}).values
        # y_train = y[:5000]
        y_train = y[:25000]
        y_train = to_categorical(y_train, 2)
        x_test = df['review_cleaned'][25000:]
        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen=max_example_len)
        y_test = y[25000:]
        y_test = to_categorical(y_test, 2)
        mixup = MIXUP()
        generator, step = mixup.flow(x_train, y_train, batch_size=batch_size, runs=1)
        if model_type == 'lstm':
            model = IMDB_LSTM()
        else:
            model = IMDB_GRU()
        model.compile('adam',
                      # "categorical_crossentropy",
                      "binary_crossentropy",
                      metrics=["accuracy"])

        checkPoint = tf.keras.callbacks.ModelCheckpoint(save_path, monitor="val_accuracy",
                                                        save_best_only=True, verbose=0)
        model.fit(generator,
                  epochs=num_epochs,
                  # batch_size=batch_size,
                  steps_per_epoch=step,
                  validation_data=(x_test, y_test),
                  callbacks=[checkPoint])
        # model.save(save_path)
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
        # print(dir(dataloader))
        mixup_generator = MixupImageDataGenerator_Iwilds(dataloader, batch_size=16)
        if model_type == 'resnet50':
            model = iwildscam_model_ResNet50()
        else:
            model = iwildscam_model_DenseNet()

        checkPoint = tf.keras.callbacks.ModelCheckpoint(save_path, monitor="val_accuracy",
                                                        save_best_only=True, verbose=1)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Adam(lr=3e-5),
                      metrics=['accuracy'])
        steps_per_epoch = 129809 // 16
        valid_per_step = 7314 // 16

        model.fit_generator(mixup_generator,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=val_dataloader,
                            validation_steps=valid_per_step,
                            epochs=10,
                            callbacks=[checkPoint])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-data",
                        type=str,
                        default='imagenet'
                        )
    parser.add_argument("--model", "-model",
                        type=str,
                        default='resnet50'
                        )
    parser.add_argument("--save", "-save",
                        type=str,
                        default='../models/imagenet/ResNet50.h5'
                        )
    args = parser.parse_args()
    data_type = args.data
    model_type = args.model
    save_path = args.save
    # data_type = 'imagenet'
    # model_type = 'resnet50'
    # save_path = "../models/imagenet/ResNet50.h5"
    mix_up_training(data_type, model_type, save_path)


# python mix-up.py -data mnist -model lenet1 -save ../models/mnist/lenet1_mixup.h5
# CUDA_VISIBLE_DEVICES=0 python mix-up.py -data imdb -model lstm -save ../models/imdb/lstm_mixup_test.h5

# python mix-up.py -data iwildscam -model resnet50 -save ../models/iwildscam/resnet50_mixup.h5
