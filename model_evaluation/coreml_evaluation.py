import coremltools
# from PIL import Image
import numpy as np
import keras
import glob
from keras.datasets import cifar10, mnist, cifar100
from keras.models import load_model
import torchvision.transforms as transforms
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import tensorflow as tf
import csv
import argparse
import sys
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from keras.preprocessing.text import Tokenizer
import keras.backend as K
import re
from wilds import get_dataset

sys.path.append('../model_prepare')
from data_prepare_tiny import *
from iWildsCam import *


def preprocessing_text(texts):
  texts = re.sub(r'<.*?>', '', texts)
  texts = re.sub(r'[^a-zA-Z]', ' ', texts)
  return ' '.join(x.lower() for x in texts.split())


def load_data(filename='./data/IMDB/IMDB_Dataset.csv'):
    '''
    :param filename: the system location of the data to load
    :return: the text (x) and its label (y)
             the text is a list of words and is not processed
    '''

    # stop words taken from nltk
    stop_words = ['i','me','my','myself','we','our','ours','ourselves','you','your','yours',
                  'yourself','yourselves','he','him','his','himself','she','her','hers','herself',
                  'it','its','itself','they','them','their','theirs','themselves','what','which',
                  'who','whom','this','that','these','those','am','is','are','was','were','be',
                  'been','being','have','has','had','having','do','does','did','doing','a','an',
                  'the','and','but','if','or','because','as','until','while','of','at','by','for',
                  'with','about','against','between','into','through','during','before','after',
                  'above','below','to','from','up','down','in','out','on','off','over','under',
                  'again','further','then','once','here','there','when','where','why','how','all',
                  'any','both','each','few','more','most','other','some','such','no','nor','not',
                  'only','own','same','so','than','too','very','s','t','can','will','just','don',
                  'should','now','d','ll','m','o','re','ve','y','ain','aren','couldn','didn',
                  'doesn','hadn','hasn','haven','isn','ma','mightn','mustn','needn','shan',
                  'shouldn','wasn','weren','won','wouldn']

    x, y = [], []
    with open(filename, "r") as f:
        for line in f:
            line = re.sub(r'\W+', ' ', line).strip().lower()  # perhaps don't make words lowercase?
            x.append(line[:-1])
            x[-1] = ' '.join(word for word in x[-1].split() if word not in stop_words)
            # print(line[-1])
            # if 'positive' in line[-1]:
            #     y.append(1)
            # else:
            #     y.append(0)
            y.append(line[-1])
    return x, np.array(y, dtype=int)


def coreml_val_main(data_type, model_type, model_path, save_path, family):
    if data_type == 'mnist':
        if family == 1:
            data_family = ["brightness"]
        elif family == 2:
            data_family = ["dotted_line"]
        elif family == 3:
            data_family = ["glass_blur"]
        elif family == 4:
            data_family = ["impulse_noise"]
        elif family == 5:
            data_family = ["rotate"]
        elif family == 6:
            data_family = ["shear"]
        elif family == 7:
            data_family = ["spatter"]
        elif family == 8:
            data_family = ["translate"]
        elif family == 9:
            data_family = ["canny_edges"]
        elif family == 10:
            data_family = ["fog"]
        elif family == 11:
            data_family = ["identity"]
        elif family == 12:
            data_family = ["motion_blur"]
        elif family == 13:
            data_family = ["scale"]
        elif family == 14:
            data_family = ["shot_noise"]
        elif family == 15:
            data_family = ["stripe"]
        elif family == 16:
            data_family = ["zigzag"]
        if model_type == 'lenet1':
            # folder_path = "../models/mnist/coreml/lenet1*mlmodel"
            # model_paths = glob.glob(folder_path)
            # print(model_paths)
            # for model_path in model_paths:
            print(model_path)
            model = coremltools.models.MLModel(model_path)
            save_labels = []
            for data_ in data_family:
                rightNum = 0
                data = np.load("../datasets/mnist_c/" + data_ + "/test_images.npy")
                labels = np.load("../datasets/mnist_c/" + data_ + "/test_labels.npy")
                data = data.astype('float32') / 255
                data = data.reshape(-1, 28, 28, 1)
                for i in range(len(data)):
                    if i % 1000 == 0:
                        print("right num: ", rightNum)
                    result = model.predict({'input_1': data[i].reshape(1, 28, 28, 1)})
                    prediction = np.argmax(result['Identity'][0])
                    save_labels.append(prediction)
                    if prediction == labels[i]:
                        rightNum += 1
                quan_acc = (rightNum / len(labels)) * 100
                print("data : {}, acc : {}".format(data_, quan_acc))
                csv_file = open(save_path, "a")
                try:
                    writer = csv.writer(csv_file)
                    writer.writerow([model_path, data_, quan_acc])
                finally:
                    csv_file.close()
                save_labels = np.asarray(save_labels)
                label_save_path = "../results/RQ3/mnist/coreml/coreml_" + model_path.split("/")[-1][:-8] + "_" + data_ + ".npy"
                np.save(label_save_path, save_labels)
        else:
            # folder_path = "../models/mnist/coreml/lenet5*mlmodel"
            # model_paths = glob.glob(folder_path)
            # for model_path in model_paths:
            model = coremltools.models.MLModel(model_path)
            print(model_path)
            save_labels = []
            for data_ in data_family:
                rightNum = 0
                data = np.load("../datasets/mnist_c/" + data_ + "/test_images.npy")
                labels = np.load("../datasets/mnist_c/" + data_ + "/test_labels.npy")
                data = data.astype('float32') / 255
                data = data.reshape(-1, 28, 28, 1)
                for i in range(len(data)):
                    if i % 1000 == 0:
                        print("right num: ", rightNum)
                    result = model.predict({'input_1': data[i].reshape(1, 28, 28, 1)})
                    prediction = np.argmax(result['Identity'][0])
                    save_labels.append(prediction)
                    if prediction == labels[i]:
                        rightNum += 1
                quan_acc = (rightNum / len(labels)) * 100
                print("data : {}, acc : {}".format(data_, quan_acc))
                csv_file = open(save_path, "a")
                try:
                    writer = csv.writer(csv_file)
                    writer.writerow([model_path, data_, quan_acc])
                finally:
                    csv_file.close()
                save_labels = np.asarray(save_labels)
                label_save_path = "../results/RQ3/mnist/coreml/coreml_" + model_path.split("/")[-1][:-8] + "_" + data_ + ".npy"
                np.save(label_save_path, save_labels)
    else:
        max_example_len = 200
        df = pd.read_csv("../datasets/IMDB/IMDB_Dataset.csv")
        df['review_cleaned'] = df['review'].apply(lambda x: preprocessing_text(x))
        x_train = df['review_cleaned'][:5000]
        tokenizer = Tokenizer(num_words=20000)
        tokenizer.fit_on_texts(x_train)

        if family == 1:
            data_ = "cr"
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
            data_ = "yelp"
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
        model = coremltools.models.MLModel(model_path)
        rightNum = 0
        save_labels = []
        for i in range(len(data)):
            if i % 1000 == 0:
                print("right num: ", rightNum)
            result = model.predict({'input_1': data[i].reshape(1, 200)})
            prediction = np.argmax(result['Identity'][0])
            save_labels.append(prediction)
            if prediction == labels[i]:
                rightNum += 1
        quan_acc = (rightNum / len(labels)) * 100
        print("data : {}, acc : {}".format(data_, quan_acc))
        csv_file = open(save_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([model_path, data_, quan_acc])
        finally:
            csv_file.close()
        save_labels = np.asarray(save_labels)
        label_save_path = "../results/RQ4/imdb/coreml/coreml_" + model_path.split("/")[-1][:-8] + "_" + data_ + ".npy"
        np.save(label_save_path, save_labels)


def coreml_val_ori(data_type, model_path, save_path):
    if data_type == 'mnist':
        (_, _), (x_test, y_test) = mnist.load_data()
        x_test = x_test.astype('float32') / 255
        x_test = x_test.reshape(-1, 28, 28, 1)

        model = coremltools.models.MLModel(model_path)
        rightNum = 0
        save_labels = []
        for i in range(len(x_test)):
            if i % 1000 == 0:
                print("right num: ", rightNum)
            result = model.predict({'input_1': x_test[i].reshape(1, 28, 28, 1)})
            prediction = np.argmax(result['Identity'][0])
            save_labels.append(prediction)
            if prediction == y_test[i]:
                rightNum += 1
        quan_acc = (rightNum / len(y_test)) * 100
        print("data : {}, acc : {}".format("ori test", quan_acc))
        csv_file = open(save_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([model_path, "oritest", quan_acc])
        finally:
            csv_file.close()
        save_labels = np.asarray(save_labels)
        label_save_path = "../results/RQ4/mnist/coreml/coreml_" + model_path.split("/")[-1][:-8] + "_oritest.npy"
        np.save(label_save_path, save_labels)
    elif data_type == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
        y_test = y_test.reshape(1, -1)[0]
        # folder_path = "../models/cifar10/tflite/*lite"
        # model_paths = glob.glob(folder_path)
        model_paths = ["../models/cifar10/coreml/nin-8bit-quan.mlmodel",
                       "../models/cifar10/coreml/resnet20-8bit-quan.mlmodel",
                       "../models/cifar10/coreml/nin-16bit-quan.mlmodel",
                       "../models/cifar10/coreml/resnet20-16bit-quan.mlmodel"
                       ]
        model = coremltools.models.MLModel(model_path)
        rightNum = 0
        save_labels = []
        for i in range(len(x_test)):
            if i % 1000 == 0:
                print("right num: ", rightNum)
            result = model.predict({'input_1': x_test[i].reshape(1, 32, 32, 3)})
            prediction = np.argmax(result['Identity'][0])
            save_labels.append(prediction)
            if prediction == y_test[i]:
                rightNum += 1
        quan_acc = (rightNum / len(y_test)) * 100
        print("data : {}, acc : {}".format("ori test", quan_acc))
        csv_file = open(save_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([model_path, "oritest", quan_acc])
        finally:
            csv_file.close()
        save_labels = np.asarray(save_labels)
        label_save_path = "../results/RQ4/cifar10/coreml/coreml_" + model_path.split("/")[-1][:-8] + "_oritest.npy"
        np.save(label_save_path, save_labels)
    elif data_type == 'imagenet':
        val_folder = '../../tiny-imagenet/tiny-imagenet-200/val/'
        data_generator = Tiny_generator_tflite(val_folder)
        # data_generator = Tiny_generator_8bit(val_folder)
        model = coremltools.models.MLModel(model_path)
        # print(model)
        right_num = 0
        total_num = 0
        batch_number = 0
        batch_total = 10000
        save_labels = np.array([])
        for x, y in data_generator:
            # print(x[0])
            y = y.argmax(axis=1)
            # print(y)
            # result = model.predict({'input_1': x.reshape(1, 224, 224, 3)}, useCPUOnly=True)
            result = model.predict({'input_1': x.reshape(1, 224, 224, 3)})
            # print(result)
            prediction = np.argmax(result['Identity'], axis=1)
            # print(prediction)
            # print(y)
            # print(wawaw)
            right_num += np.sum(y == prediction)
            total_num += len(x)
            if batch_number % 100 == 0:
                print("right / total: {} / {}".format(right_num, total_num))
            save_labels = np.concatenate((save_labels, prediction))
            batch_number += 1
            if batch_number == batch_total:
                break
            # break
        quan_acc = (right_num / total_num) * 100
        acc_save_path = save_path
        csv_file = open(acc_save_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([model_path, "oritest", quan_acc])
        finally:
            csv_file.close()
        label_save_path = "../results/RQ3/imagenet/coreml_" + model_path.split("/")[-1][:-8] + "_oritest.npy"
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
        y_train = tf.keras.utils.to_categorical(y_train, 2)
        x_test = df['review_cleaned'][25000:30000]
        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen=max_example_len)
        y_test = y[25000:30000]
        model = coremltools.models.MLModel(model_path)
        rightNum = 0
        save_labels = []
        for i in range(len(x_test)):
            if i % 1000 == 0:
                print("right num: ", rightNum)
            result = model.predict({'input_1': x_test[i].reshape(1, 200)})
            prediction = np.argmax(result['Identity'][0])
            save_labels.append(prediction)
            if prediction == y_test[i]:
                rightNum += 1
        quan_acc = (rightNum / len(y_test)) * 100
        print("data : {}, acc : {}".format("oritest", quan_acc))
        csv_file = open(save_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([model_path, "oritest", quan_acc])
        finally:
            csv_file.close()
        save_labels = np.asarray(save_labels)
        label_save_path = "../results/RQ4/imdb/coreml/coreml_" + model_path.split("/")[-1][:-8] + "_oritest.npy"
        np.save(label_save_path, save_labels)
    else:
        dataset = get_dataset(dataset='iwildcam', download=True)
        val_data = dataset.get_subset('id_test',
                                      transform=transforms.Compose([transforms.Resize((448, 448)),
                                                                    transforms.ToTensor()]))
        val_loader = get_eval_loader('standard', val_data, batch_size=1)
        data_generator = DataGenerator(val_loader, 182)
        model = coremltools.models.MLModel(model_path)
        batch_number = 0
        right_num = 0
        total_num = 0
        batch_total = 8154
        save_labels = np.array([])
        for x, y in data_generator:
            y = y.argmax()
            # print(y)
            result = model.predict({'input_1': x.reshape(1, 448, 448, 3)})
            # print(result['Identity'])
            prediction = np.argmax(result['Identity'], axis=1)
            # print(prediction)
            right_num += np.sum(y == prediction)
            total_num += len(x)
            if batch_number % 200 == 0:
                print("right / total: {} / {}".format(right_num, total_num))
            save_labels = np.concatenate((save_labels, prediction))
            batch_number += 1
            if batch_number == batch_total:
                break
            # break
        quan_acc = (right_num / total_num) * 100
        acc_save_path = save_path
        csv_file = open(acc_save_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([model_path, "oritest", quan_acc])
        finally:
            csv_file.close()
        label_save_path = "../results/RQ3/wilds/coreml/coreml_" + model_path.split("/")[-1][:-8] + "_oritest.npy"
        np.save(label_save_path, save_labels)


def coreml_val_main_imagenet(data_type, data_path, model_path, save_path, label_save_path):
    if data_type == 'imagenet':
        data_path_final = '../../../Downloads/Tiny-ImageNet-C/' + data_path + '/1/'
        # data_generator = Tiny_generator_tflite(data_path_final)
        data_generator = Tiny_generator_8bit(data_path_final)
        model = coremltools.models.MLModel(model_path)
        # print(model)
        right_num = 0
        total_num = 0
        batch_number = 0
        batch_total = 10000
        save_labels = np.array([])
        for x, y in data_generator:
            # print(x[0])
            y = y.argmax(axis=1)
            # print(y)
            # result = model.predict({'input_1': x.reshape(1, 224, 224, 3)}, useCPUOnly=True)
            result = model.predict({'input_1': x.reshape(1, 224, 224, 3)})
            # print(result)
            prediction = np.argmax(result['Identity'], axis=1)
            # print(prediction)
            # print(y)
            # print(wawaw)
            right_num += np.sum(y == prediction)
            total_num += len(x)
            if batch_number % 100 == 0:
                print("right / total: {} / {}".format(right_num, total_num))
            save_labels = np.concatenate((save_labels, prediction))
            batch_number += 1
            if batch_number == batch_total:
                break
            # break
        quan_acc = (right_num / total_num) * 100
        acc_save_path = save_path
        csv_file = open(acc_save_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([model_path, data_path, quan_acc])
        finally:
            csv_file.close()
        np.save(label_save_path, save_labels)

    else:
        dataset = get_dataset(dataset='iwildcam', download=True)
        val_data = dataset.get_subset('test',
                                      transform=transforms.Compose([transforms.Resize((448, 448)),
                                                                    transforms.ToTensor()]))
        val_loader = get_eval_loader('standard', val_data, batch_size=1)
        data_generator = DataGenerator(val_loader, 182)
        model = coremltools.models.MLModel(model_path)
        batch_number = 0
        right_num = 0
        total_num = 0
        batch_total = 42791
        save_labels = np.array([])
        for x, y in data_generator:
            y = y.argmax()
            # print(y)
            result = model.predict({'input_1': x.reshape(1, 448, 448, 3)})
            # print(result['Identity'])
            prediction = np.argmax(result['Identity'], axis=1)
            # print(prediction)
            right_num += np.sum(y == prediction)
            total_num += len(x)
            if batch_number % 200 == 0:
                print("right / total: {} / {}".format(right_num, total_num))
            save_labels = np.concatenate((save_labels, prediction))
            batch_number += 1
            if batch_number == batch_total:
                break
            # break
        quan_acc = (right_num / total_num) * 100
        acc_save_path = save_path
        csv_file = open(acc_save_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([model_path, data_path, quan_acc])
        finally:
            csv_file.close()
        np.save(label_save_path, save_labels)


def coreml_val_main_cifar10(data_path, model_path, save_path, label_save_path, severity=1):
    (x_train, _), (_, _) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_train_mean = np.mean(x_train, axis=0)
    del x_train
    # corrupted data prepare
    data_path_final = '../../../Downloads/CIFAR-10-C/' + data_path + ".npy"
    index_start = int(10000 * (severity - 1))
    index_end = index_start + 10000
    test_images = np.load(data_path_final)[index_start: index_end]
    test_images = test_images.astype('float32') / 255
    test_images -= x_train_mean
    labels = np.load('../../../Downloads/CIFAR-10-C/labels.npy')[index_start: index_end]
    model = coremltools.models.MLModel(model_path)
    split_len = 1
    save_labels = np.array([])
    for i in range(10000):
        if i % 100 == 0:
            print("batch: ", i)
            # break
        # batch_image = test_images[int(i * split_len): int((i + 1) * split_len)]
        batch_image = test_images[i]
        result = model.predict({'input_1': batch_image.reshape(1, 32, 32, 3)})
        prediction = np.argmax(result['Identity'], axis=1)
        save_labels = np.concatenate((save_labels, prediction))
    save_labels = save_labels.astype(np.uint8)
    print(len(labels))
    print(len(save_labels))
    print(np.sum(labels == save_labels))
    quan_acc = (np.sum(labels == save_labels) * 100) / len(labels)
    print("data : {}, acc : {}".format(data_path, quan_acc))
    csv_file = open(save_path, "a")
    try:
        writer = csv.writer(csv_file)
        writer.writerow([model_path, data_path, quan_acc])
    finally:
        csv_file.close()
    np.save(label_save_path, save_labels)


def mnist_run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", "-data_type",
                        type=str,
                        default='mnist'
                        )
    parser.add_argument("--model_type", "-model_type",
                        type=str,
                        default='lenet1'
                        )
    parser.add_argument("--model_path", "-model_path",
                        type=str,
                        default='../models/mnist/coreml/lenet1_qa-4bits-quan.mlmodel'
                        )
    parser.add_argument("--save_path", "-save_path",
                        type=str,
                        default='../results/RQ1/lenet1_acc.h5'
                        )
    parser.add_argument("--family", "-family",
                        type=int,
                        default=1
                        )

    args = parser.parse_args()
    data_type = args.data_type
    model_type = args.model_type
    model_path = args.model_path
    save_path = args.save_path
    family = args.family
    coreml_val_main(data_type, model_type, model_path, save_path, family)


def cifar10_run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-data_path",
                        type=str,
                        default='brightness',
                        )
    parser.add_argument("--model_path", "-model_path",
                        type=str,
                        default='../models/cifar10/coreml/nin-16bit-quan.mlmodel'
                        )
    parser.add_argument("--save_path", "-save_path",
                        type=str,
                        default='../results/RQ1/coreml_cifar10_nin_acc.csv'
                        )
    parser.add_argument("--label_save_path", "-label_save_path",
                        type=str,
                        default='../results/RQ3/cifar10/coreml_nin_16bit_brightness.npy'
                        )
    parser.add_argument("--severity", "-severity",
                        type=int,
                        default=1
                        )

    args = parser.parse_args()
    data_path = args.data_path
    model_path = args.model_path
    save_path = args.save_path
    label_save_path = args.label_save_path
    severity = args.severity
    coreml_val_main_cifar10(data_path, model_path, save_path, label_save_path, severity)


def imagenet_run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", "-data_type",
                        type=str,
                        default='imagenet'
                        )
    parser.add_argument("--data_path", "-data_path",
                        type=str,
                        default='brightness'
                        )
    parser.add_argument("--model_path", "-model_path",
                        type=str,
                        default='../models/imagenet/coreml/resnet101-8bit-quan.mlmodel'
                        )
    parser.add_argument("--save_path", "-save_path",
                        type=str,
                        default='../results/RQ1/coreml_imagenet_acc.csv'
                        )
    parser.add_argument("--label_save_path", "-label_save_path",
                        type=str,
                        default='../results/RQ3/imagenet/coreml/resnet101-8bit-quan_brightness.npy'
                        )

    args = parser.parse_args()
    data_type = args.data_type
    data_path = args.data_path
    model_path = args.model_path
    save_path = args.save_path
    label_save_path = args.label_save_path
    coreml_val_main_imagenet(data_type, data_path, model_path, save_path, label_save_path)


if __name__ == '__main__':
    # mnist_run()
    # imagenet_run()
    # cifar10_run()
    # model_paths = [
    #                "../models/mnist/coreml/lenet1_adv-8bits-quan.mlmodel",
    #                "../models/mnist/coreml/lenet1_adv-16bits-quan.mlmodel",
    #                "../models/mnist/coreml/lenet1_mixup-8bits-quan.mlmodel",
    #                "../models/mnist/coreml/lenet1_mixup-16bits-quan.mlmodel",
    #                "../models/mnist/coreml/lenet1_qa-8bits-quan.mlmodel",
    #                "../models/mnist/coreml/lenet1_qa-16bits-quan.mlmodel",
    #                 "../models/mnist/coreml/lenet5_adv-8bits-quan.mlmodel",
    #                 "../models/mnist/coreml/lenet5_adv-16bits-quan.mlmodel",
    #                 "../models/mnist/coreml/lenet5_mixup-8bits-quan.mlmodel",
    #                 "../models/mnist/coreml/lenet5_mixup-16bits-quan.mlmodel",
    #                 "../models/mnist/coreml/lenet5_qa-8bits-quan.mlmodel",
    #                 "../models/mnist/coreml/lenet5_qa-16bits-quan.mlmodel"
    # ]
    # model_paths = ["../models/cifar10/coreml/nin_adv-8bit-quan.mlmodel",
    #                "../models/cifar10/coreml/nin_qa-8bit-quan.mlmodel",
    #                "../models/cifar10/coreml/nin_mixup-8bit-quan.mlmodel",
    #                "../models/cifar10/coreml/nin_adv-16bit-quan.mlmodel",
    #                "../models/cifar10/coreml/nin_qa-16bit-quan.mlmodel",
    #                "../models/cifar10/coreml/nin_mixup-16bit-quan.mlmodel",
    #                "../models/cifar10/coreml/resnet20_adv-8bit-quan.mlmodel",
    #                "../models/cifar10/coreml/resnet20_qa-8bit-quan.mlmodel",
    #                "../models/cifar10/coreml/resnet20_mixup-8bit-quan.mlmodel",
    #                "../models/cifar10/coreml/resnet20_adv-16bit-quan.mlmodel",
    #                "../models/cifar10/coreml/resnet20_qa-16bit-quan.mlmodel",
    #                "../models/cifar10/coreml/resnet20_mixup-16bit-quan.mlmodel",
    #                ]
    # model_paths = [
    #                "../models/imagenet/coreml/densenet-8bit-quan.mlmodel",
    #                "../models/imagenet/coreml/densenet-16bit-quan.mlmodel",
    #                "../models/imagenet/coreml/resnet101-8bit-quan.mlmodel",
    #                # "../models/imagenet/coreml/resnet101-16bit-quan.mlmodel"
    # ]

    # model_paths = [
        # "../models/imdb/coreml/lstm-8bits-quan.mlmodel",
        # "../models/imdb/coreml/lstm-16bits-quan.mlmodel",
        # "../models/imdb/coreml/lstm_ag-8bits-quan.mlmodel",
        # "../models/imdb/coreml/lstm_ag-16bits-quan.mlmodel",
        # "../models/imdb/coreml/lstm_adv-8bits-quan.mlmodel",
        # "../models/imdb/coreml/lstm_adv-16bits-quan.mlmodel",
        # "../models/imdb/coreml/lstm_mixup-8bits-quan.mlmodel",
        # "../models/imdb/coreml/lstm_mixup-16bits-quan.mlmodel",
        # "../models/imdb/coreml/gru-8bits-quan.mlmodel",
        # "../models/imdb/coreml/gru-16bits-quan.mlmodel",
        # "../models/imdb/coreml/gru_ag-8bits-quan.mlmodel",
        # "../models/imdb/coreml/gru_ag-16bits-quan.mlmodel",
        # "../models/imdb/coreml/gru-8bits-quan_adv.mlmodel",
        # "../models/imdb/coreml/gru-16bits-quan_adv.mlmodel",
        # "../models/imdb/coreml/gru_mixup-8bits-quan.mlmodel",
        # "../models/imdb/coreml/gru_mixup-16bits-quan.mlmodel",
    # ]
    # model_paths = [
    #     # "../models/iwildscam/coreml/densenet-8bit-quan.mlmodel",
    #     # "../models/iwildscam/coreml/densenet-16bit-quan.mlmodel",
    #     # "../models/iwildscam/coreml/resnet50-8bit-quan.mlmodel",
    #     # "../models/iwildscam/coreml/resnet50-16bit-quan.mlmodel",
    #     # "../models/iwildscam/coreml/densenet_adv-8bit-quan.mlmodel",
    #     # "../models/iwildscam/coreml/densenet_adv-16bit-quan.mlmodel",
    #     # "../models/iwildscam/coreml/densenet_da-8bit-quan.mlmodel",
    #     # "../models/iwildscam/coreml/densenet_da-16bit-quan.mlmodel",
    #     # "../models/iwildscam/coreml/densenet_mixup-8bit-quan.mlmodel",
    #     # "../models/iwildscam/coreml/densenet_mixup-16bit-quan.mlmodel",
    #     # "../models/iwildscam/coreml/densenet_qa-8bit-quan.mlmodel",
    #     # "../models/iwildscam/coreml/densenet_qa-16bit-quan.mlmodel",
    #     # "../models/iwildscam/coreml/resnet50_adv-8bit-quan.mlmodel",
    #     # "../models/iwildscam/coreml/resnet50_adv-16bit-quan.mlmodel",
    #     # "../models/iwildscam/coreml/resnet50_da-8bit-quan.mlmodel",
    #     # "../models/iwildscam/coreml/resnet50_da-16bit-quan.mlmodel",
    #     # "../models/iwildscam/coreml/resnet50_mixup-8bit-quan.mlmodel",
    #     # "../models/iwildscam/coreml/resnet50_mixup-16bit-quan.mlmodel",
    #     # "../models/iwildscam/coreml/resnet50_qa-8bit-quan.mlmodel",
    #     # "../models/iwildscam/coreml/resnet50_qa-16bit-quan.mlmodel"
    #     "../models/iwildscam/coreml/densenet_adv2_8bits.mlmodel",
    #     "../models/iwildscam/coreml/densenet_adv2_16bits.mlmodel",
    #     "../models/iwildscam/coreml/resnet50_adv2_8bits.mlmodel",
    #     "../models/iwildscam/coreml/resnet50_adv2_16bits.mlmodel"
    #
    # ]
    model_paths = [
        "../models/RQ4/cifar10/coreml/nin_8bits.mlmodel",
        "../models/RQ4/cifar10/coreml/nin_16bits.mlmodel",
        "../models/RQ4/cifar10/coreml/resnet20_8bits.mlmodel",
        "../models/RQ4/cifar10/coreml/resnet20_16bits.mlmodel"
    ]
    for model_path in model_paths:
        coreml_val_ori("cifar10", model_path, "../results/RQ4/cifar10/coreml_cifar10_acc.csv")

    model_paths = [
        "../models/RQ4/imdb/coreml/gru_8bits.mlmodel",
        "../models/RQ4/imdb/coreml/gru_16bits.mlmodel",
        "../models/RQ4/imdb/coreml/lstm_8bits.mlmodel",
        "../models/RQ4/imdb/coreml/lstm_16bits.mlmodel"
    ]
    for model_path in model_paths:
        coreml_val_ori("imdb", model_path, "../results/RQ4/imdb/coreml_imdb_acc.csv")

    model_paths = [
        "../models/RQ4/wilds/coreml/densenet_8bits.mlmodel",
        "../models/RQ4/wilds/coreml/densenet_16bits.mlmodel",
        "../models/RQ4/wilds/coreml/resnet50_8bits.mlmodel",
        "../models/RQ4/wilds/coreml/resnet50_16bits.mlmodel"
    ]
    for model_path in model_paths:
        coreml_val_ori("wilds", model_path, "../results/RQ4/wilds/coreml_wilds_acc.csv")

    # for model_path in model_paths:
    #     coreml_val_main_imagenet("wilds", "OOD", model_path, "../results/RQ1/coreml_wilds_acc.csv", "../results/RQ3/wilds/coreml/" + model_path.split("/")[-1][:-8] + "_OOD.npy")
    # coreml_val_ori("cifar10", "../models/imagenet/coreml/resnet101-16bit-quan.mlmodel", "../results/RQ1/coreml_imagenet_acc.csv")

# python coreml_evaluation.py -data_type mnist -model_type lenet1 -model_path $path -save_path ../results/RQ1/coreml_lenet1_acc.csv -family 1
# python coreml_evaluation.py -data_type imdb -model_type lstm -model_path ../models/imdb/coreml/gru-8bits-quan.mlmodel -save_path ../results/RQ1/coreml_lstm_acc.h5 -family 1
# python coreml_evaluation.py -data_type imagenet -data_path brightness -model_path ../models/imagenet/coreml/resnet101_adv-8bit-quan.mlmodel -save_path "../results/RQ1/coreml_imagenet_acc.csv" -label_save_path ../results/RQ3/imagenet/coreml/resnet101_adv-8bit-quan_brightness.npy
# python coreml_evaluation.py -data_type imagenet -data_path brightness -model_path ../models/imagenet/coreml/resnet101-8bit-quan.mlmodel -save_path "../results/RQ1/coreml_imagenet_acc.csv" -label_save_path ../results/RQ3/imagenet/coreml/resnet101-8bit-quan_brightness.npy
