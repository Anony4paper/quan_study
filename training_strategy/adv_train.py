"""
This is an example of how to use ART and Keras to perform adversarial training using data generators for CIFAR10
"""
# import keras
from tensorflow.keras.datasets import mnist
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import KerasClassifier
from art.defences.trainer import AdversarialTrainer
import tensorflow as tf
import argparse
import sys
import torch

import tensorflow_addons as tfa
from wilds import get_dataset
from art.data_generators import KerasDataGenerator
from textattack import AttackArgs, Attacker
from textattack.attack_recipes import PWWSRen2019
import textattack
from tensorflow.keras.utils import to_categorical
# from textattack.models.wrappers import ModelWrapper
import numpy as np
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.keras.models import Model, Sequential
import foolbox

sys.path.append('../model_prepare')
from imagenet_models import *
from IMDB import *
from iWildsCam import *
from LeNet import *
from Newsgroups import *
from data_prepare_tiny import *
from ResNet20 import *
from NiN import *

tf.compat.v1.disable_eager_execution()


def schedule(epoch_idx):
    if (epoch_idx + 1) < 10:
        return 1e-03
    elif (epoch_idx + 1) < 20:
        return 1e-04  # lr_decay_ratio = 0.2
    elif (epoch_idx + 1) < 30:
        return 1e-05
    return 1e-05


def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(training_config)
        )
    restored_model.set_weights(weights)
    return restored_model


def make_keras_picklable():
    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__


class CustomKerasModelWrapper(ModelWrapper):

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text_input_list):
        # x_transform = []
        # for i, review in enumerate(text_input_list):
        #     # tokens = [x.strip(",") for x in review.split()]
        #     # BoW_array = np.zeros((NUM_WORDS,))
        #     # for word in tokens:
        #     #     if word in vocabulary:
        #     #         if vocabulary[word] < len(BoW_array):
        #     #             BoW_array[vocabulary[word]] += 1
        #     self.tokenizertokenizer.texts_to_sequences
        #     x_transform.append(BoW_array)
        x_transform = self.tokenizer.texts_to_sequences(text_input_list)
        x_transform = tf.keras.preprocessing.sequence.pad_sequences(x_transform, maxlen=200)
        x_transform = np.array(x_transform)
        prediction = self.model.predict(x_transform)
        # print("wawawawawawaw")
        # print(prediction)
        return prediction


def at_text(model, x, y, x_test, y_test, tokenizer, batch_size, save_path, max_example_len=200, epoch=10, label_num=2):
    torch.multiprocessing.freeze_support()
    make_keras_picklable()
    model_wrapper = CustomKerasModelWrapper(model, tokenizer)
    x = x.to_numpy()
    # y = y.to_numpy()
    batch_num = len(x) // batch_size
    attack = PWWSRen2019.build(model_wrapper)

    num_per_batch = int(batch_size / 2)
    attack_args = AttackArgs(
        num_examples=num_per_batch,
        checkpoint_dir="checkpoints",
        parallel=True,
        num_workers_per_device=10,
        silent=True,
        disable_stdout=True
    )
    data_len = len(x)
    save_acc = 0
    for i in range(epoch):
        # random_index = np.random.choice(data_len, data_len, replace=False)
        # x = x[random_index]
        # y = y[random_index]
        for j in range(batch_num):
            print("start: ", j * batch_size)
            print("end: ", j * batch_size + batch_size)

            batch_x = x[j * batch_size: j * batch_size + batch_size]
            batch_y = y[j * batch_size: j * batch_size + batch_size]

            # choose candidate data
            selected_index = np.random.choice(batch_size, num_per_batch, replace=False)
            adv_batch_x = batch_x[selected_index]
            adv_batch_y = batch_y[selected_index]

            my_dataset = []
            for m in range(num_per_batch):
                my_dataset.append((adv_batch_x[m], int(adv_batch_y[m])))
            # print(my_dataset)
            new_dataset = textattack.datasets.Dataset(my_dataset)
            attacker = Attacker(attack, new_dataset, attack_args)
            results = attacker.attack_dataset()
            adv_data = []
            # print(results)
            for _ in results:
                adv_data.append(tokenizer.texts_to_sequences([_.perturbed_text()])[0])
            # print(adv_data)
            adv_data = pad_sequences(adv_data, maxlen=max_example_len)
            batch_x = tokenizer.texts_to_sequences(batch_x)
            batch_x = pad_sequences(batch_x, maxlen=max_example_len)
            batch_x[selected_index] = adv_data
            batch_y = to_categorical(batch_y, label_num)

            model.fit(batch_x, batch_y, batch_size=batch_size, epochs=1)
            del attacker
        _, acc = model.evaluate(x_test, y_test)
        if acc > save_acc:
            model.save(save_path)
            save_acc = acc


def at_img(model, training_generator, val_generator, num_epoch, batch_size, save_path):
    fmodel = foolbox.TensorFlowModel(model, bounds=(0, 1))
    attack = foolbox.attacks.LinfProjectedGradientDescentAttack(abs_stepsize=2/255, steps=4, random_start=True)
    # attack = foolbox.attacks.LinfFastGradientAttack(random_start=True)
    epsilons = 8 / 255
    save_acc = 0
    for i in range(num_epoch):
        batch_num = 0
        if i == 10:
            fmodel._model.compile(loss='categorical_crossentropy',
                                  optimizer=tfa.optimizers.SGDW(lr=1e-04, momentum=0.9, weight_decay=1e-4),
                                  # optimizer=optimizers.Adam(lr=1e-03),
                                  metrics=['accuracy'])
        if i == 20:
            fmodel._model.compile(loss='categorical_crossentropy',
                                  optimizer=tfa.optimizers.SGDW(lr=1e-05, momentum=0.9, weight_decay=1e-4),
                                  # optimizer=optimizers.Adam(lr=1e-03),
                                  metrics=['accuracy'])
        for x, y in training_generator:
            print("batch number: ", batch_num)
            batch_num += 1
            labels = np.argmax(y, axis=1)
            images = tf.convert_to_tensor(x)
            labels = tf.convert_to_tensor(labels, dtype=tf.int32)
            _, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)
            fmodel._model.fit(clipped_advs.numpy(), y, batch_size=batch_size, epochs=1)

        _, acc = fmodel._model.evaluate_generator(val_generator,
                                                  10000 // batch_size,
                                                  workers=12)
        print("epoch: ", i)
        print("val acc: ", acc)
        if acc > save_acc:
            fmodel._model.save(save_path)
            save_acc = acc


def adv_train(data_type, model_type, save_path):
    if data_type == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
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
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      metrics=['accuracy'])

        classifier = KerasClassifier(model, clip_values=(0, 1), use_logits=False)
        pgd = ProjectedGradientDescent(classifier, eps=0.3, eps_step=0.1, max_iter=10, num_random_init=1)
        adv_trainer = AdversarialTrainer(classifier, attacks=pgd, ratio=0.5)
        # print(dir(adv_trainer.get_classifier().model))
        adv_trainer.fit(x_train, y_train, batch_size=batch_size, nb_epochs=epochs)
        adv_trainer.get_classifier().model.save(save_path)
        score = adv_trainer.get_classifier().model.evaluate(x_test, y_test)
        print(score)
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
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                      metrics=['accuracy'])
        classifier = KerasClassifier(model, clip_values=(0, 1), use_logits=False)
        pgd = ProjectedGradientDescent(classifier, eps=4 / 255, eps_step=1 / 255, max_iter=4, num_random_init=1)
        adv_trainer = AdversarialTrainer(classifier, attacks=pgd, ratio=0.5)
        adv_trainer.fit(x_train, y_train, batch_size=batch_size, nb_epochs=epochs)
        adv_trainer.get_classifier().model.save(save_path)
        score = adv_trainer.get_classifier().model.evaluate(x_test, y_test)
        print(score)
    elif data_type == 'imagenet':
        training_folder = '../datasets/tiny-imagenet/tiny-imagenet-200/train/'
        val_folder = '../datasets/tiny-imagenet/tiny-imagenet-200/val/'
        base_train_generator, base_val_generator = Tiny_generator(training_folder, val_folder)

        train_generator = KerasDataGenerator(
            base_train_generator,
            size=100000,
            batch_size=64
        )

        val_generator = KerasDataGenerator(
            base_val_generator,
            size=10000,
            batch_size=64
        )
        config = imagenet_config()
        config.set_defaults()
        # checkPoint = tf.keras.callbacks.ModelCheckpoint(save_path, monitor="val_accuracy",
        #                                                 save_best_only=True, verbose=0)
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(schedule=schedule)
        # cbs = [lr_schedule, checkPoint]
        cbs = [lr_schedule]
        if model_type == 'resnet50':
            model = ResNet50_model()
        elif model_type == 'resnet101':
            model = tf.keras.models.load_model("../models/imagenet/resnet101_adv_50.h5")
            save_acc = 68.28
            # model = ResNet101_model()
        elif model_type == 'efficientnet':
            model = Efficientnet_model()
        else:
            model = tf.keras.models.load_model("../models/imagenet/densenet_adv_50.h5")
            save_acc = 63.19
            # model = DenseNet_model()
        # model.summary()
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(loss='categorical_crossentropy',
                      optimizer=tfa.optimizers.SGDW(lr=1e-03, momentum=0.9, weight_decay=1e-4),
                      # optimizer=optimizers.Adam(lr=1e-03),
                      metrics=['accuracy'])
        # at_img(model, base_train_generator, base_val_generator, config.epochs, config.batch_size, save_path)
        classifier = KerasClassifier(model, clip_values=(0, 1), use_logits=False)
        pgd = ProjectedGradientDescent(classifier, eps=4/255, eps_step=1/255, max_iter=4, num_random_init=1)
        adv_trainer = AdversarialTrainer(classifier, attacks=pgd, ratio=0.3)

        # save_acc = 0
        epoch_num = 0
        total_epoch = 50
        for _ in range(total_epoch):
            adv_trainer.fit_generator(generator=train_generator,
                                      nb_epochs=1,
                                      callbacks=cbs
                                      )

            _, acc = adv_trainer.get_classifier().model.evaluate_generator(base_val_generator,
                                                                           10000 // config.batch_size,
                                                                           workers=12)
            print(acc)
            if acc > save_acc:
                adv_trainer.get_classifier().model.save(save_path)
                save_acc = acc
            # score = adv_trainer.get_classifier().model.save(save_path)

    elif data_type == 'iwildscam':
        dataset = get_dataset(dataset='iwildcam', download=True, root_dir="/scratch/users/qihu/data")
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

        train_generator = KerasDataGenerator(
            dataloader,
            size=129809,
            batch_size=16
        )

        # val_dataloader = KerasDataGenerator(
        #     val_dataloader,
        #     size=7314,
        #     batch_size=16
        # )

        if model_type == 'resnet50':
            # model = iwildscam_model_ResNet50()
            model = tf.keras.models.load_model("../models/iwildscam/resnet50_adv.h5")
        else:
            # model = iwildscam_model_DenseNet()
            model = tf.keras.models.load_model("../models/iwildscam/densenet_adv.h5")
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Adam(lr=3e-5),
                      metrics=['accuracy'])
        steps_per_epoch = 129809 // 16
        valid_per_step = 7314 // 16
        classifier = KerasClassifier(model, clip_values=(0, 1), use_logits=False)
        pgd = ProjectedGradientDescent(classifier, eps=4 / 255, eps_step=1 / 255, max_iter=4, num_random_init=1)
        adv_trainer = AdversarialTrainer(classifier, attacks=pgd, ratio=0.3)
        save_acc = 0
        total_epoch = 20
        for _ in range(total_epoch):
            adv_trainer.fit_generator(generator=train_generator,
                                      nb_epochs=1,
                                      # steps_per_epoch=100000 // config.batch_size,
                                      # validation_data=val_generator,
                                      # validation_steps=10000 // config.batch_size,
                                      # use_multiprocessing=True,
                                      # callbacks=cbs
                                      )
            _, acc = adv_trainer.get_classifier().model.evaluate_generator(val_dataloader,
                                                                           valid_per_step,
                                                                           use_multiprocessing=True,
                                                                           workers=12)
            print(acc)
            if acc > save_acc:
                adv_trainer.get_classifier().model.save(save_path)
                save_acc = acc
        # adv_trainer.get_classifier().model.save(save_path)
        # score = model.evaluate_generator(val_dataloader, valid_per_step, workers=12)
        # print(score)
    elif data_type == 'imdb':
        max_example_len = 200
        batch_size = 32
        embedding_dims = 50
        vocab_size = 19999
        num_epochs = 10
        label_num = 2
        df = pd.read_csv("../datasets/IMDB/IMDB_Dataset.csv")
        df['review_cleaned'] = df['review'].apply(lambda x: preprocessing_text(x))
        x_train = df['review_cleaned'][:5000]
        tokenizer = Tokenizer(num_words=20000)
        tokenizer.fit_on_texts(x_train)
        y = df['sentiment'].map({'negative': 0, 'positive': 1}).values
        y_train = y[:5000]
        x_test = df['review_cleaned'][25000:30000]
        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen=max_example_len)
        y_test = y[25000:30000]
        y_test = to_categorical(y_test, 2)
        if model_type == 'lstm':
            model = IMDB_LSTM()
        else:
            model = IMDB_GRU()
        model.compile("adam",
                      "categorical_crossentropy",
                      metrics=["accuracy"])
        at_text(model,
                x_train,
                y_train,
                x_test,
                y_test,
                tokenizer,
                batch_size,
                save_path,
                max_example_len=max_example_len,
                epoch=num_epochs,
                label_num=label_num)


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
    # model_path = "models/lenet1.h5"
    # save_path = "models/lenet1_adv.h5"
    # adv_train_old(model_path, save_path)
    adv_train(data_type, model_type, save_path)

# CUDA_VISIBLE_DEVICES=0 python adv_train.py -data imdb -model lstm -save ../models/imdb/lstm_adv.h5
# CUDA_VISIBLE_DEVICES=0 python adv_train.py -data imagenet -model resnet101 -save ../models/imagenet/resnet101_adv.h5
