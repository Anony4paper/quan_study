import tensorflow as tf
import numpy as np
import csv
from keras.datasets import cifar10, mnist
from tensorflow_addons.optimizers import SGDW
import tensorflow_model_optimization as tfmot
tf.keras.optimizers.SGDW = SGDW
import sys
from coreml_evaluation import *
sys.path.append('../model_prepare')
from data_prepare_tiny import *
from iWildsCam import *
from IMDB import *


def model_eval_main(data_type, save_path):
    if data_type == 'mnist':
        model_base = "../models/mnist/"
        model_paths = ["lenet1.h5", "lenet1_adv.h5", "lenet1_ag.h5", "lenet1_mixup.h5", "lenet1_qa.h5",
                       "lenet5.h5", "lenet5_adv.h5", "lenet5_ag.h5", "lenet5_mixup.h5", "lenet5_qa.h5"]
        data_family = ["brightness", "canny_edges", "dotted_line", "fog", "glass_blur", "identity",
                       "impulse_noise", "motion_blur", "rotate", "scale", "shear", "shot_noise",
                       "spatter", "stripe", "translate", "zigzag"]
        for data_ in data_family:
            data = np.load("../datasets/mnist_c/" + data_ + "/test_images.npy")
            labels = np.load("../datasets/mnist_c/" + data_ + "/test_labels.npy")
            data = data.astype('float32') / 255
            data = data.reshape(-1, 28, 28, 1)
            for model_path in model_paths:
                model_path_final = model_base + model_path
                if "_qa" in model_path_final:
                    with tfmot.quantization.keras.quantize_scope():
                        model = tf.keras.models.load_model(model_path_final)
                else:
                    model = tf.keras.models.load_model(model_path_final)
                predictions = model.predict(data, batch_size=256)
                predict_labels = predictions.argmax(axis=1)
                # print(labels)
                # print(predict_labels)
                acc = (np.sum(labels == predict_labels) * 100) / len(labels)
                print("data : {}, acc : {}".format(data_, acc))
                csv_file = open(save_path, "a")
                try:
                    writer = csv.writer(csv_file)
                    writer.writerow([model_path, data_, acc])
                finally:
                    csv_file.close()
                label_save_base = "../results/RQ3/mnist/ori/"
                label_save_path = label_save_base + model_path.split(".")[0] + "_" + data_ + ".npy"
                np.save(label_save_path, predict_labels)
            #     break
            # break
    elif data_type == "cifar10":
        model_base = "../models/cifar10/"
        model_paths = ["nin.h5", "nin_adv.h5", "nin_da.h5", "nin_mixup.h5", "nin_qa.h5",
                       "resnet20.h5", "resnet20_adv.h5", "resnet20_da.h5", "resnet20_mixup.h5", "resnet20_qa.h5"]
        data_family = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
                       "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression",
                       "motion_blur", "pixelate", "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"]

        for data_ in data_family:
            # data = np.load('../../../Downloads/CIFAR-10-C/' + data_ + '.npy')[:10000]
            # labels = np.load('../../../Downloads/CIFAR-10-C/labels.npy')[:10000]

            data = np.load('/scratch/users/qihu/CIFAR-10-C/' + data_ + '.npy')[:10000]
            labels = np.load('/scratch/users/qihu/CIFAR-10-C/labels.npy')[:10000]
            (x_train, _), (_, _) = cifar10.load_data()
            x_train = x_train.astype('float32') / 255
            x_train_mean = np.mean(x_train, axis=0)
            data = data.astype('float32') / 255
            data -= x_train_mean
            for model_path in model_paths:
                model_path_final = model_base + model_path
                if "_qa" in model_path_final:
                    with tfmot.quantization.keras.quantize_scope():
                        model = tf.keras.models.load_model(model_path_final)
                else:
                    model = tf.keras.models.load_model(model_path_final)
                predictions = model.predict(data, batch_size=256)
                predict_labels = predictions.argmax(axis=1)
                acc = (np.sum(labels == predict_labels) * 100) / len(labels)
                print("data : {}, acc : {}".format(data_, acc))
                csv_file = open(save_path, "a")
                try:
                    writer = csv.writer(csv_file)
                    writer.writerow([model_path, data_, acc])
                finally:
                    csv_file.close()
                label_save_base = "../results/RQ3/cifar10/ori/"
                label_save_path = label_save_base + model_path.split(".")[0] + "_" + data_ + ".npy"
                np.save(label_save_path, predict_labels)
    elif data_type == 'imagenet':
        val_folder = '../datasets/tiny-imagenet/tiny-imagenet-200/val/'
        train_generator, val_generator = Tiny_generator(val_folder, val_folder)
        model_base = "../models/imagenet/"
        model_paths = ["densenet.h5", "densenet_adv.h5", "densenet_da.h5", "densenet_mixup.h5", "densenet_qa_real.h5",
                       "resnet101.h5", "resnet101_adv.h5", "resnet101_da.h5", "resnet101_mixup.h5", "resnet101_qa_real.h5"]
        data_family = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_noise",
                       "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate", "shot_noise",
                       "snow", "zoom_blur"]

        for model_path in model_paths:
            model_path_final = model_base + model_path
            if "_qa" in model_path_final:
                with tfmot.quantization.keras.quantize_scope():
                    model = tf.keras.models.load_model(model_path_final)
            else:
                model = tf.keras.models.load_model(model_path_final)

            for data_ in data_family:
                # data_path_final = '../../../Downloads/Tiny-ImageNet-C/' + data_ + '/1/'
                data_path_final = '../datasets/tiny-imagenet/Tiny-ImageNet-C/' + data_ + '/1/'
                data_generator = Tiny_generator_tflite(data_path_final)
                right_num = 0
                total_num = 0
                batch_number = 0
                batch_total = 200
                save_labels = np.array([])
                for x, y in data_generator:
                    y = y.argmax(axis=1)
                    predictions = model.predict(x).argmax(axis=1)
                    right_num += np.sum(y == predictions)
                    total_num += len(x)
                    print("right / total: {} / {}".format(right_num, total_num))
                    save_labels = np.concatenate((save_labels, predictions))
                    batch_number += 1
                    if batch_number == batch_total:
                        break
                quan_acc = (right_num / total_num) * 100
                acc_save_path = save_path
                csv_file = open(acc_save_path, "a")
                try:
                    writer = csv.writer(csv_file)
                    writer.writerow([model_path, data_, quan_acc])
                finally:
                    csv_file.close()
                label_save_base = "../results/RQ3/imagenet/ori/"
                label_save_path = label_save_base + model_path.split(".")[0] + "_" + data_ + ".npy"
                np.save(label_save_path, save_labels)
    elif data_type == 'imdb':
        max_example_len = 200
        df = pd.read_csv("../datasets/IMDB/IMDB_Dataset.csv")
        df['review_cleaned'] = df['review'].apply(lambda x: preprocessing_text(x))
        x_train = df['review_cleaned'][:5000]
        tokenizer = Tokenizer(num_words=20000)
        tokenizer.fit_on_texts(x_train)
        data_family = ["cr", "yelp"]
        model_paths = ["lstm_5000.h5", "lstm_adv.h5", "lstm_ag.h5", "lstm_mixup.h5", "lstm_qa.h5",
                       "gru_5000.h5", "gru_adv.h5", "gru_ag.h5", "gru_mixup.h5"]
        model_base = "../models/imdb/"
        for data_ in data_family:
            if data_ == "cr":
                cr_data, cr_labels = load_data('../datasets/IMDB/CR.train')
                cr_data = tokenizer.texts_to_sequences(cr_data)
                cr_data = pad_sequences(cr_data, maxlen=max_example_len)

                cr_data_dev, cr_labels_dev = load_data('../datasets/IMDB/CR.dev')
                cr_data_dev = tokenizer.texts_to_sequences(cr_data_dev)
                cr_data_dev = pad_sequences(cr_data_dev, maxlen=max_example_len)

                cr_data_test, cr_labels_test = load_data('../datasets/IMDB/CR.test')
                cr_data_test = tokenizer.texts_to_sequences(cr_data_test)
                cr_data_test = pad_sequences(cr_data_test, maxlen=max_example_len)

                rt_data, rt_labels = load_data('../datasets/IMDB/MR.train')
                rt_data = tokenizer.texts_to_sequences(rt_data)
                rt_data = pad_sequences(rt_data, maxlen=max_example_len)

                rt_data_dev, rt_labels_dev = load_data('../datasets/IMDB/MR.dev')
                rt_data_dev = tokenizer.texts_to_sequences(rt_data_dev)
                rt_data_dev = pad_sequences(rt_data_dev, maxlen=max_example_len)

                rt_data_test, rt_labels_test = load_data('../datasets/IMDB/MR.test')
                rt_data_test = tokenizer.texts_to_sequences(rt_data_test)
                rt_data_test = pad_sequences(rt_data_test, maxlen=max_example_len)

                ood_data = np.concatenate((cr_data, cr_data_dev))
                ood_data = np.concatenate((ood_data, cr_data_test))
                ood_label = np.concatenate((cr_labels, cr_labels_dev))
                ood_label = np.concatenate((ood_label, cr_labels_test))

                ood_data = np.concatenate((ood_data, rt_data))
                ood_data = np.concatenate((ood_data, rt_data_dev))
                ood_data = np.concatenate((ood_data, rt_data_test))

                ood_label = np.concatenate((ood_label, rt_labels))
                ood_label = np.concatenate((ood_label, rt_labels_dev))
                ood_label = np.concatenate((ood_label, rt_labels_test))
                data = ood_data[:5000]
                labels = ood_label[:5000]
            else:
                yelp_reviews = pd.read_csv("../datasets/IMDB/yelp_academic_dataset_review.csv",
                                           usecols=["stars", "text"],
                                           nrows=100000)
                yelp_reviews.head(5)

                reviews = yelp_reviews[:]
                reviews = reviews[reviews.stars != 3]

                reviews["labels"] = reviews["stars"].apply(lambda x: 1 if x > 3 else 0)
                reviews = reviews.drop("stars", axis=1)
                texts = reviews["text"].values
                labels = reviews["labels"].values
                # print(texts[0])
                yelp_data = tokenizer.texts_to_sequences(texts)
                yelp_data = pad_sequences(yelp_data, maxlen=max_example_len)
                data = yelp_data[:5000]
                labels = labels[:5000]
            for model_path in model_paths:
                model_path_final = model_base + model_path
                if "_qa" in model_path_final:
                    with tfmot.quantization.keras.quantize_scope():
                        model = tf.keras.models.load_model(model_path_final)
                else:
                    model = tf.keras.models.load_model(model_path_final)
                predictions = model.predict(data, batch_size=256)
                predict_labels = predictions.argmax(axis=1)
                # print(labels)
                # print(predict_labels)
                acc = (np.sum(labels == predict_labels) * 100) / len(labels)
                print("data : {}, acc : {}".format(data_, acc))
                csv_file = open(save_path, "a")
                try:
                    writer = csv.writer(csv_file)
                    writer.writerow([model_path, data_, acc])
                finally:
                    csv_file.close()
                label_save_base = "../results/RQ3/imdb/ori/"
                label_save_path = label_save_base + model_path.split(".")[0] + "_" + data_ + ".npy"
                np.save(label_save_path, predict_labels)
    else:
        dataset = get_dataset(dataset='iwildcam', download=False)
        val_data = dataset.get_subset('test',
                                      transform=transforms.Compose([transforms.Resize((448, 448)),
                                                                    transforms.ToTensor()]))
        val_loader = get_eval_loader('standard', val_data, batch_size=7)
        data_generator = DataGenerator(val_loader, 182)
        model_paths = [
            # "../models/iwildscam/densenet.h5",
            # "../models/iwildscam/resnet50.h5",
            # "../models/iwildscam/densenet_adv.h5",
            # "../models/iwildscam/densenet_da.h5",
            # "../models/iwildscam/densenet_mixup.h5",
            # "../models/iwildscam/densenet_qa.h5",
            # "../models/iwildscam/resnet50_adv.h5",
            # "../models/iwildscam/resnet50_da.h5",
            # "../models/iwildscam/resnet50_mixup.h5",
            # "../models/iwildscam/resnet50_qa.h5"
            "../models/iwildscam/resnet50_adv_2.h5",
            "../models/iwildscam/densenet_adv_2.h5"
        ]
        for model_path in model_paths:
            if "_qa" in model_path:
                with tfmot.quantization.keras.quantize_scope():
                    model = tf.keras.models.load_model(model_path)
            else:
                model = tf.keras.models.load_model(model_path)
            batch_number = 0
            right_num = 0
            total_num = 0
            # batch_total = 42791
            batch_total = 6113
            # save_labels = []
            save_labels = np.array([])
            for x, y in data_generator:
                y = y.argmax(axis=1)
                prediction = np.argmax(model.predict(x), axis=1)
                # print(y.shape)
                # print(prediction.shape)
                right_num += np.sum(y == prediction)
                total_num += len(x)
                if batch_number % 100 == 0:
                    print("right / total: {} / {}".format(right_num, total_num))
                save_labels = np.concatenate((save_labels, prediction))
                # save_labels.append(prediction)
                batch_number += 1
                if batch_number == batch_total:
                    break
                # break
            save_labels = np.asarray(save_labels)
            acc = (right_num / total_num) * 100
            acc_save_path = save_path
            csv_file = open(acc_save_path, "a")
            try:
                writer = csv.writer(csv_file)
                writer.writerow([model_path, "OOD", acc])
            finally:
                csv_file.close()
            label_save_base = "../results/RQ3/wilds/ori/"
            label_save_path = label_save_base + model_path.split("/")[-1].split(".")[0] + "_OOD.npy"
            np.save(label_save_path, save_labels)


def model_eval_ori_main(data_type, save_path):
    if data_type == 'mnist':
        (_, _), (x_test, y_test) = mnist.load_data()
        x_test = x_test.astype('float32') / 255
        x_test = x_test.reshape(-1, 28, 28, 1)

        model_paths = [
                       # "../models/mnist/lenet1.h5",
                       # "../models/mnist/lenet1_adv.h5",
                       # "../models/mnist/lenet1_ag.h5",
                       # "../models/mnist/lenet1_mixup.h5",
                       # "../models/mnist/lenet1_qa.h5",
                       # "../models/mnist/lenet5.h5",
                       # "../models/mnist/lenet5_adv.h5",
                       # "../models/mnist/lenet5_ag.h5",
                       # "../models/mnist/lenet5_mixup.h5",
                       # "../models/mnist/lenet5_qa.h5"
                        "../models/RQ4/mnist/lenet5.h5"
                       ]
        for model_path in model_paths:
            if "_qa" in model_path:
                with tfmot.quantization.keras.quantize_scope():
                    model = tf.keras.models.load_model(model_path)
            else:
                model = tf.keras.models.load_model(model_path)
            predictions = model.predict(x_test, batch_size=256)
            predict_labels = predictions.argmax(axis=1)
            acc = (np.sum(y_test == predict_labels) * 100) / len(y_test)
            print("model : {}, acc : {}".format(model_path, acc))
            csv_file = open(save_path, "a")
            try:
                writer = csv.writer(csv_file)
                writer.writerow([model_path, "ori_test", acc])
            finally:
                csv_file.close()
            label_save_base = "../results/RQ4/mnist/ori/"
            label_save_path = label_save_base + model_path.split("/")[-1].split(".")[0] + "_oritest.npy"
            np.save(label_save_path, predict_labels)
    elif data_type == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
        y_test = y_test.reshape(1, -1)[0]
        # model_paths = ["../models/cifar10/nin.h5", "../models/cifar10/resnet20.h5"]
        # model_paths = ["../models/cifar10/nin_mixup.h5", "../models/cifar10/resnet20_mixup.h5",
        #                "../models/cifar10/nin_qa.h5", "../models/cifar10/resnet20_qa.h5",
        #                "../models/cifar10/nin_adv.h5", "../models/cifar10/resnet20_adv.h5"]
        model_paths = ["../models/RQ4/cifar10/nin.h5", "../models/RQ4/cifar10/resnet20.h5"]
        for model_path in model_paths:
            if "_qa" in model_path:
                with tfmot.quantization.keras.quantize_scope():
                    model = tf.keras.models.load_model(model_path)
            else:
                model = tf.keras.models.load_model(model_path)
            predictions = model.predict(x_test, batch_size=256)
            predict_labels = predictions.argmax(axis=1)
            # print(y_test.shape)
            # print(predict_labels)
            acc = (np.sum(y_test == predict_labels) * 100) / len(y_test)
            print("model : {}, acc : {}".format(model_path, acc))
            csv_file = open(save_path, "a")
            try:
                writer = csv.writer(csv_file)
                writer.writerow([model_path, "ori_test", acc])
            finally:
                csv_file.close()
            label_save_base = "../results/RQ4/cifar10/ori/"
            label_save_path = label_save_base + model_path.split("/")[-1].split(".")[0] + "_oritest.npy"
            np.save(label_save_path, predict_labels)
    elif data_type == 'imagenet':
        val_folder = '../datasets/tiny-imagenet/tiny-imagenet-200/val/'
        data_generator = Tiny_generator_tflite(val_folder)
        model_paths = [
                       "../models/imagenet/densenet.h5",
                       # "../models/imagenet/densenet_adv_50.h5",
                       # "../models/imagenet/densenet_ag.h5",
                       # "../models/imagenet/densenet_mixup.h5",
                       # "../models/imagenet/densenet_qa.h5",
                       "../models/imagenet/resnet101.h5",
                       # "../models/imagenet/resnet101_adv_50.h5",
                       # "../models/imagenet/resnet101_ag.h5",
                       # "../models/imagenet/resnet101_mixup.h5",
                       # "../models/imagenet/resnet101_qa.h5"
                       ]

        for model_path in model_paths:
            if "_qa" in model_path:
                with tfmot.quantization.keras.quantize_scope():
                    model = tf.keras.models.load_model(model_path)
            else:
                model = tf.keras.models.load_model(model_path)
            right_num = 0
            total_num = 0
            batch_number = 0
            batch_total = 200
            save_labels = np.array([])
            for x, y in data_generator:
                y = y.argmax(axis=1)
                predictions = model.predict(x).argmax(axis=1)
                right_num += np.sum(y == predictions)
                total_num += len(x)
                print("right / total: {} / {}".format(right_num, total_num))
                save_labels = np.concatenate((save_labels, predictions))
                batch_number += 1
                if batch_number == batch_total:
                    break
            quan_acc = (right_num / total_num) * 100
            acc_save_path = save_path
            csv_file = open(acc_save_path, "a")
            try:
                writer = csv.writer(csv_file)
                writer.writerow([model_path, "ori_test", quan_acc])
            finally:
                csv_file.close()
            label_save_base = "../results/RQ3/imagenet/ori/"
            label_save_path = label_save_base + model_path.split("/")[-1].split(".")[0] + "_oritest.npy"
            np.save(label_save_path, save_labels)
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
        x_train = pad_sequences(x_train, maxlen=max_example_len)
        y = df['sentiment'].map({'negative': 0, 'positive': 1}).values
        y_train = y[:5000]
        y_train = to_categorical(y_train, 2)
        x_test = df['review_cleaned'][25000:30000]
        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen=max_example_len)
        y_test = y[25000:30000]
        # y_test = to_categorical(y_test, 2)

        model_paths = [
            # "../models/imdb/lstm_5000.h5",
            # "../models/imdb/lstm_adv.h5",
            # "../models/imdb/lstm_ag.h5",
            # "../models/imdb/lstm_mixup.h5",
            # "../models/imdb/lstm_qa.h5",
            # "../models/imdb/gru_5000.h5",
            # "../models/imdb/gru_adv.h5",
            # "../models/imdb/gru_ag.h5",
            # "../models/imdb/gru_mixup.h5",
            # "../models/imdb/lenet5_qa.h5"
            "../models/RQ4/imdb/lstm.h5",
            "../models/RQ4/imdb/gru.h5"
        ]
        for model_path in model_paths:
            if "_qa" in model_path:
                with tfmot.quantization.keras.quantize_scope():
                    model = tf.keras.models.load_model(model_path)
            else:
                model = tf.keras.models.load_model(model_path)
            predictions = model.predict(x_test, batch_size=256)
            predict_labels = predictions.argmax(axis=1)
            acc = (np.sum(y_test == predict_labels) * 100) / len(y_test)
            print("model : {}, acc : {}".format(model_path, acc))
            csv_file = open(save_path, "a")
            try:
                writer = csv.writer(csv_file)
                writer.writerow([model_path, "ori_test", acc])
            finally:
                csv_file.close()
            label_save_base = "../results/RQ4/imdb/ori/"
            label_save_path = label_save_base + model_path.split("/")[-1].split(".")[0] + "_oritest.npy"
            np.save(label_save_path, predict_labels)
    else:
        dataset = get_dataset(dataset='iwildcam', download=False)
        val_data = dataset.get_subset('id_test',
                                      transform=transforms.Compose([transforms.Resize((448, 448)),
                                                                    transforms.ToTensor()]))
        val_loader = get_eval_loader('standard', val_data, batch_size=1)
        data_generator = DataGenerator(val_loader, 182)
        model_paths = [
            # "../models/iwildscam/densenet.h5",
            # "../models/iwildscam/densenet_adv.h5",
            # "../models/iwildscam/densenet_da.h5",
            # "../models/iwildscam/densenet_mixup.h5",
            # "../models/iwildscam/densenet_qa.h5",
            # "../models/iwildscam/resnet50.h5",
            # "../models/iwildscam/resnet50_adv.h5",
            # "../models/iwildscam/resnet50_da.h5",
            # "../models/iwildscam/resnet50_mixup.h5",
            # "../models/iwildscam/resnet50_adv_2.h5",
            # "../models/iwildscam/densenet_adv_2.h5"
            "../models/RQ4/wilds/densenet.h5",
            "../models/RQ4/wilds/resnet50.h5"
        ]
        for model_path in model_paths:
            if "_qa" in model_path:
                with tfmot.quantization.keras.quantize_scope():
                    model = tf.keras.models.load_model(model_path)
            else:
                model = tf.keras.models.load_model(model_path)
            batch_number = 0
            right_num = 0
            total_num = 0
            batch_total = 8154
            save_labels = []
            for x, y in data_generator:
                y = y.argmax()
                prediction = np.argmax(model.predict(x))
                # print(y.shape)
                # print(prediction.shape)
                right_num += np.sum(y == prediction)
                total_num += len(x)
                if batch_number % 200 == 0:
                    print("right / total: {} / {}".format(right_num, total_num))
                # save_labels = np.concatenate((save_labels, prediction))
                save_labels.append(prediction)
                batch_number += 1
                if batch_number == batch_total:
                    break
                # break
            save_labels = np.asarray(save_labels)
            acc = (right_num / total_num) * 100
            acc_save_path = save_path
            csv_file = open(acc_save_path, "a")
            try:
                writer = csv.writer(csv_file)
                writer.writerow([model_path, "ori_test", acc])
            finally:
                csv_file.close()
            label_save_base = "../results/RQ4/wilds/ori/"
            label_save_path = label_save_base + model_path.split("/")[-1].split(".")[0] + "_oritest.npy"
            np.save(label_save_path, save_labels)


if __name__ == "__main__":
    # # # mnist
    # data_type = "imagenet"
    # save_path = "../results/RQ1/ori_imagenet.csv"
    # # model_eval_main(data_type, save_path)
    # model_eval_ori_main(data_type, save_path)

    # # cifar10
    # data_type = "cifar10"
    # save_path = "../results/RQ4/cifar10/ori_cifar10.csv"
    # # model_eval_main(data_type, save_path)
    # model_eval_ori_main(data_type, save_path)

    # imdb
    # data_type = "imdb"
    # save_path = "../results/RQ4/imdb/ori_imdb.csv"
    # model_eval_ori_main(data_type, save_path)

    #  mnist
    # data_type = "mnist"
    # save_path = "../results/RQ4/mnist/ori_mnist.csv"
    # model_eval_ori_main(data_type, save_path)

#     wilds
    data_type = "wilds"
    save_path = "../results/RQ4/wilds/ori_wilds.csv"
    # model_eval_main(data_type, save_path)
    model_eval_ori_main(data_type, save_path)
