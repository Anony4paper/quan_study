import tensorflow as tf
import keras
import numpy as np
import glob
import csv
import argparse
import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.datasets import cifar10
import torchvision.transforms as transforms
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds import get_dataset
import sys
from keras.datasets import cifar10, mnist
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


def load_cps_model_new(path):
    interpreter = tf.lite.Interpreter(model_path=path)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print(input_details)
    # Test the model on random input data.
    # input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    def predict(inputs):
        outputs = []
        # inputs = inputs.astype(np.float32)
        # inputs = inputs.astype(np.int8)
        # print(inputs.shape)
        for i in range(0, inputs.shape[0]):
            input_data = inputs[i:i+1, ::]
            # input_data = inputs[i]

            input_data = input_data.reshape(1, 200)
            # print(input_data.shape)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
            output_data = interpreter.get_tensor(output_details[0]['index'])
            outputs.append(output_data)
        return np.vstack(outputs)

    return interpreter, predict


def run_tflite_model(tflite_file, test_image_indices, test_images):

  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(model_path=str(tflite_file))

  input_details = interpreter.get_input_details()
  # IMDB
  interpreter.resize_tensor_input(
    input_details[0]['index'], (1, 200))
  # image only
  interpreter.allocate_tensors()
  # interpreter.resizeInputShape(1, 200)

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  predictions = np.zeros((len(test_image_indices),), dtype=int)
  for i, test_image_index in enumerate(test_image_indices):
    if i % 1000 == 0:
      print(i)
    test_image = test_images[test_image_index]

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8 or input_details['dtype'] == np.float16:
      input_scale, input_zero_point = input_details["quantization"]
      # print(input_scale)
      test_image = test_image / input_scale + input_zero_point
      # print("haha")
      # print(test_image)
      # return

    # test_image = test_image.reshape(1, 200)
    # print(test_image.shape)
    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    # print(test_image.shape)
    # input_details = interpreter.get_input_details()
    # print(input_details)
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    # print(output.argmax())
    predictions[i] = output.argmax()

  return predictions


def run_tflite_model_imagenet(interpreter, test_image_indices, test_images):

  # Initialize the interpreter
  # interpreter = tf.lite.Interpreter(model_path=str(tflite_file))

  input_details = interpreter.get_input_details()
  # imagenet
  # interpreter.resize_tensor_input(
  #   input_details[0]['index'], (1, 224, 224, 3))
  # iWildsCam
  interpreter.resize_tensor_input(
      input_details[0]['index'], (1, 448, 448, 3))
  # image only
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]
  print("test1")
  predictions = np.zeros((len(test_image_indices),), dtype=int)
  for i, test_image_index in enumerate(test_image_indices):
    test_image = test_images[test_image_index]
    # print(i)
    print("test2")
    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8 or input_details['dtype'] == np.float16:

      input_scale, input_zero_point = input_details["quantization"]
      print(input_scale)
      print(input_zero_point)
      test_image = test_image / input_scale + input_zero_point
    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_image)
    print("test3")
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    # print(output.argmax())
    print("test4")
    predictions[i] = output.argmax()

  return predictions


def run_tflite_model_cifar10(interpreter, test_image_indices, test_images):

  # Initialize the interpreter
  # interpreter = tf.lite.Interpreter(model_path=str(tflite_file))

  input_details = interpreter.get_input_details()
  # IMDB
  interpreter.resize_tensor_input(
    input_details[0]['index'], (1, 32, 32, 3))
  # image only
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  predictions = np.zeros((len(test_image_indices),), dtype=int)
  for i, test_image_index in enumerate(test_image_indices):
    test_image = test_images[test_image_index]
    if i % 100 == 0:
        print(i)
    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8 or input_details['dtype'] == np.float16:

      input_scale, input_zero_point = input_details["quantization"]
      # print(input_scale)
      # print(input_zero_point)
      test_image = test_image / input_scale + input_zero_point
    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    # print(output.argmax())
    predictions[i] = output.argmax()
  return predictions


def run_tflite_model_imagenet_batch_test(interpreter, test_image_indices, test_images):

  # Initialize the interpreter
  # interpreter = tf.lite.Interpreter(model_path=str(tflite_file))

    input_details = interpreter.get_input_details()
    # IMDB
    interpreter.resize_tensor_input(
    input_details[0]['index'], (100, 224, 224, 3))
    # image only
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # predictions = np.zeros((len(test_image_indices),), dtype=int)
    # for i, test_image_index in enumerate(test_image_indices):
    #     test_image = test_images[test_image_index]
    #     print(i)
    #     # Check if the input type is quantized, then rescale input data to uint8
    #     if input_details['dtype'] == np.uint8 or input_details['dtype'] == np.float16:
    #         input_scale, input_zero_point = input_details["quantization"]
    #         test_image = test_image / input_scale + input_zero_point
    #     test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    #     interpreter.set_tensor(input_details["index"], test_image)
    #     interpreter.invoke()
    #     output = interpreter.get_tensor(output_details["index"])[0]
    #     # print(output.argmax())
    #     predictions[i] = output.argmax()

    if input_details['dtype'] == np.uint8 or input_details['dtype'] == np.float16:
        input_scale, input_zero_point = input_details["quantization"]
        # print(input_scale)
        # print(input_zero_point)
        test_images = test_images / input_scale + input_zero_point

    # test_images = np.expand_dims(test_images, axis=0).astype(input_details["dtype"])

    test_images = test_images.astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_images)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])
    predictions = output.argmax(axis=1)
    return predictions


def ori_model_evaluation(model, data, label):
    # print(data.shape)
    predictions = model.predict(data)
    predicted_labels = np.argmax(predictions, axis=1)
    # print(predicted_labels)
    right_num = len(np.where(predicted_labels == label)[0])
    # print("right number: ", right_num)
    # print("acc: ", right_num / len(label))
    return right_num / len(label)


def compressed_model_evaluation(model, data, label):
    predictions = model(data)
    predicted_labels = np.argmax(predictions, axis=1)
    right_num = len(np.where(predicted_labels == label)[0])
    # print("right number: ", right_num)
    # print("acc: ", right_num / len(label))
    return right_num / len(label)


def tflite_val_main(data_type, model_type, save_path):
    if data_type == 'mnist':
        data_family = ["brightness", "canny_edges", "dotted_line", "fog", "glass_blur", "identity",
                       "impulse_noise", "motion_blur", "rotate", "scale", "shear", "shot_noise",
                       "spatter", "stripe", "translate", "zigzag"]
        if model_type == 'lenet1':
            folder_path = "../models/mnist/tflite/lenet1*lite"
            model_paths = glob.glob(folder_path)
            for model_path in model_paths:
                _, cps_model = load_cps_model_new(model_path)
                for data_ in data_family:
                    data = np.load("../datasets/mnist_c/" + data_ + "/test_images.npy")
                    labels = np.load("../datasets/mnist_c/" + data_ + "/test_labels.npy")
                    data = data.astype('float32') / 255
                    data = data.reshape(-1, 28, 28, 1)
                    test_image_indices = range(data.shape[0])
                    predictions = run_tflite_model(model_path, test_image_indices, data)
                    quan_acc = (np.sum(labels == predictions) * 100) / len(labels)
                    # quan_acc = compressed_model_evaluation(cps_model, data, labels)
                    print("data : {}, acc : {}".format(data_, quan_acc))
                    csv_file = open(save_path, "a")
                    try:
                        writer = csv.writer(csv_file)
                        writer.writerow([model_path, data_, quan_acc])
                    finally:
                        csv_file.close()
                    label_save_path = "../results/RQ3/mnist/" + model_path.split("/")[-1][:-5] + "_" + data_ + ".npy"
                    np.save(label_save_path, predictions)

        else:
            folder_path = "../models/mnist/tflite/lenet5*lite"
            model_paths = glob.glob(folder_path)
            for model_path in model_paths:
                _, cps_model = load_cps_model_new(model_path)
                for data_ in data_family:
                    data = np.load("../datasets/mnist_c/" + data_ + "/test_images.npy")
                    labels = np.load("../datasets/mnist_c/" + data_ + "/test_labels.npy")
                    data = data.astype('float32') / 255
                    data = data.reshape(-1, 28, 28, 1)
                    test_image_indices = range(data.shape[0])
                    predictions = run_tflite_model(model_path, test_image_indices, data)
                    quan_acc = (np.sum(labels == predictions) * 100) / len(labels)
                    # quan_acc = compressed_model_evaluation(cps_model, data, labels)
                    print("data : {}, acc : {}".format(data_, quan_acc))
                    csv_file = open(save_path, "a")
                    try:
                        writer = csv.writer(csv_file)
                        writer.writerow([model_path, quan_acc])
                    finally:
                        csv_file.close()
                    label_save_path = "../results/RQ3/mnist/" + model_path.split("/")[-1][:-5] + "_" + data_ + ".npy"
                    np.save(label_save_path, predictions)
    else:
        max_example_len = 200
        df = pd.read_csv("../datasets/IMDB/IMDB_Dataset.csv")
        df['review_cleaned'] = df['review'].apply(lambda x: preprocessing_text(x))
        x_train = df['review_cleaned'][:5000]
        tokenizer = Tokenizer(num_words=20000)
        tokenizer.fit_on_texts(x_train)
        data_family = ['cr', 'yelp']
        for data_ in data_family:
            if data_ == 'yelp':
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
            else:
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

            if model_type == 'lstm':
                folder_path = "../models/imdb/tflite/lstm*lite"
                model_paths = glob.glob(folder_path)
                for model_path in model_paths:
                    print(model_path)
                    _, cps_model = load_cps_model_new(model_path)
                    test_image_indices = range(data.shape[0])
                    predictions = run_tflite_model(model_path, test_image_indices, data)
                    quan_acc = (np.sum(labels == predictions) * 100) / len(labels)
                    # quan_acc = compressed_model_evaluation(cps_model, data, labels)
                    print("data : {}, acc : {}".format(data_, quan_acc))
                    csv_file = open(save_path, "a")
                    try:
                        writer = csv.writer(csv_file)
                        writer.writerow([model_path, data_, quan_acc])
                    finally:
                        csv_file.close()
                    label_save_path = "../results/RQ3/imdb/" + model_path.split("/")[-1][:-5] + "_" + data_ + ".npy"
                    np.save(label_save_path, predictions)
            else:
                folder_path = "../models/imdb/tflite/gru*lite"
                model_paths = glob.glob(folder_path)
                for model_path in model_paths:
                    print(model_path)
                    _, cps_model = load_cps_model_new(model_path)
                    test_image_indices = range(data.shape[0])
                    predictions = run_tflite_model(model_path, test_image_indices, data)
                    quan_acc = (np.sum(labels == predictions) * 100) / len(labels)
                    # quan_acc = compressed_model_evaluation(cps_model, data, labels)
                    print("data : {}, acc : {}".format(data_, quan_acc))
                    csv_file = open(save_path, "a")
                    try:
                        writer = csv.writer(csv_file)
                        writer.writerow([model_path, data_, quan_acc])
                    finally:
                        csv_file.close()
                    label_save_path = "../results/RQ3/imdb/" + model_path.split("/")[-1][:-5] + "_" + data_ + ".npy"
                    np.save(label_save_path, predictions)


def tflite_val_main_ori(data_type, save_path):
    if data_type == 'mnist':
        # folder_path = "../models/mnist/tflite/*lite"
        # model_paths = glob.glob(folder_path)
        model_paths = [
            "../models/RQ4/mnist/lenet5_8bits.lite",
            "../models/RQ4/mnist/lenet5_16bits.lite"
        ]
        (_, _), (x_test, y_test) = mnist.load_data()
        x_test = x_test.astype('float32') / 255
        x_test = x_test.reshape(-1, 28, 28, 1)
        for model_path in model_paths:
            _, cps_model = load_cps_model_new(model_path)
            test_image_indices = range(x_test.shape[0])
            predictions = run_tflite_model(model_path, test_image_indices, x_test)
            quan_acc = (np.sum(y_test == predictions) * 100) / len(y_test)
            # quan_acc = compressed_model_evaluation(cps_model, data, labels)
            print("data : {}, acc : {}".format("ori test", quan_acc))
            csv_file = open(save_path, "a")
            try:
                writer = csv.writer(csv_file)
                writer.writerow([model_path, "ori_test", quan_acc])
            finally:
                csv_file.close()
            label_save_path = "../results/RQ4/mnist/tflite/" + model_path.split("/")[-1][:-5] + "_oritest.npy"
            np.save(label_save_path, predictions)
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
        model_paths = ["../models/cifar10/tflite/nin-8bits-quan.lite",
                       "../models/cifar10/tflite/resnet20-8bits-quan.lite",]
        for model_path in model_paths:
            _, cps_model = load_cps_model_new(model_path)
            test_image_indices = range(x_test.shape[0])
            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            predictions = run_tflite_model_cifar10(interpreter, test_image_indices, x_test)
            quan_acc = (np.sum(y_test == predictions) * 100) / len(y_test)
            # quan_acc = compressed_model_evaluation(cps_model, data, labels)
            print("data : {}, acc : {}".format("ori test", quan_acc))
            csv_file = open(save_path, "a")
            try:
                writer = csv.writer(csv_file)
                writer.writerow([model_path, "ori_test", quan_acc])
            finally:
                csv_file.close()
            label_save_path = "../results/RQ3/cifar10/tflite/" + model_path.split("/")[-1][:-5] + "_oritest.npy"
            np.save(label_save_path, predictions)
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
        model_paths = [
            # "../models/imdb/tflite/lstm-16bits-quan.lite",
            # "../models/imdb/tflite/lstm_ag-16bits-quan.lite",
            # "../models/imdb/tflite/lstm_adv-16bits-quan.lite",
            # "../models/imdb/tflite/lstm_mixup-16bits-quan.lite",
            # "../models/imdb/tflite/gru-16bits-quan.lite",
            # "../models/imdb/tflite/gru-16bits-quan_adv.lite",
            # "../models/imdb/tflite/gru_ag-16bits-quan.lite",
            # "../models/imdb/tflite/gru_mixup-16bits-quan.lite"
            "../models/RQ4/imdb/tflite/gru_16bits.lite",
            "../models/RQ4/imdb/tflite/lstm_16bits.lite"
        ]
        for model_path in model_paths:
            _, cps_model = load_cps_model_new(model_path)
            test_image_indices = range(x_test.shape[0])
            predictions = run_tflite_model(model_path, test_image_indices, x_test)
            quan_acc = (np.sum(y_test == predictions) * 100) / len(y_test)
            # quan_acc = compressed_model_evaluation(cps_model, data, labels)
            print("data : {}, acc : {}".format("ori test", quan_acc))
            csv_file = open(save_path, "a")
            try:
                writer = csv.writer(csv_file)
                writer.writerow([model_path, "ori_test", quan_acc])
            finally:
                csv_file.close()
            label_save_path = "../results/RQ4/imdb/tflite/" + model_path.split("/")[-1][:-5] + "_oritest.npy"
            np.save(label_save_path, predictions)



def tflite_val_main_imagenet(data_type, data_path, model_path, save_path):
    if data_type == 'imagenet':
        # data_path = "../datasets/tiny-imagenet/Tiny-ImageNet-C/" + data_path + "/1/"
        data_path = '../../../Downloads/Tiny-ImageNet-C/' + data_path + '/1/'
        data_generator = Tiny_generator_tflite(data_path)
        # _, cps_model = load_cps_model_new(model_path)

        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        right_num = 0
        total_num = 0
        batch_number = 0
        batch_total = 200
        for x, y in data_generator:
            # print("batch: ", batch_number)
            y = y.argmax(axis=1)
            # print(y)
            test_image_indices = range(x.shape[0])
            predictions = run_tflite_model_imagenet(interpreter, test_image_indices, x)
            right_num += np.sum(y == predictions)
            total_num += len(x)
            print("right / total: {} / {}".format(right_num, total_num))

            batch_number += 1
            if batch_number == batch_total:
                break

        print("finally")
        print("acc: ", (right_num / total_num) * 100)
        csv_file = open(save_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([model_path, data_path, (right_num / total_num) * 100])
        finally:
            csv_file.close()
    else:
        dataset = get_dataset(dataset='iwildcam', download=True)
        val_data = dataset.get_subset('id_test',
                                      transform=transforms.Compose([transforms.Resize((448, 448)),
                                                                    transforms.ToTensor()]))
        val_loader = get_eval_loader('standard', val_data, batch_size=1)
        data_generator = DataGenerator(val_loader, 182)
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        batch_number = 0
        right_num = 0
        total_num = 0
        batch_total = 8154
        save_labels = np.array([])
        for x, y in data_generator:
            # print("batch: ", batch_number)
            y = y.argmax(axis=1)
            # print(y)
            test_image_indices = range(x.shape[0])
            predictions = run_tflite_model_imagenet(interpreter, test_image_indices, x)
            save_labels = np.concatenate((save_labels, predictions))
            right_num += np.sum(y == predictions)
            total_num += len(x)
            print("right / total: {} / {}".format(right_num, total_num))

            batch_number += 1
            if batch_number == batch_total:
                break
        print("finally")
        print("acc: ", (right_num / total_num) * 100)
        csv_file = open(save_path, "a")
        try:
            writer = csv.writer(csv_file)
            writer.writerow([model_path, data_path, (right_num / total_num) * 100])
        finally:
            csv_file.close()

        label_save_path = "../results/RQ3/wilds/" + model_path.split("/")[-1][:-5] + "_oritest.npy"
        np.save(label_save_path, save_labels)


def tflite_val_main_cifar10(data_path, model_path, save_path, label_save_path, severity=1):
    (x_train, _), (_, _) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_train_mean = np.mean(x_train, axis=0)
    del x_train
    # corrupted data prepare
    # data_path_final = "/scratch/users/qihu/CIFAR-10-C/" + data_path + ".npy"
    data_path_final = "../datasets/cifar10/CIFAR-10-C/" + data_path + ".npy"
    # data_path_final = '../../../Downloads/CIFAR-10-C/' + data_path + ".npy"
    index_start = int(10000 * (severity - 1))
    index_end = index_start + 10000
    test_images = np.load(data_path_final)[index_start: index_end]
    test_images = test_images.astype('float32') / 255
    test_images -= x_train_mean
    # print(test_images.shape)
    # labels = np.load('/scratch/users/qihu/CIFAR-10-C/labels.npy')[index_start: index_end]
    labels = np.load('../datasets/cifar10/CIFAR-10-C/labels.npy')[index_start: index_end]
    # tflite model
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    test_image_indices = range(test_images.shape[0])
    total_predictions = run_tflite_model_cifar10(interpreter, test_image_indices, test_images)
    quan_acc = (np.sum(labels == total_predictions) * 100) / len(labels)
    # print(labels)
    # print(total_predictions)
    #
    # input_details = interpreter.get_input_details()
    # interpreter.resize_tensor_input(input_details[0]['index'], (50, 32, 32, 3))
    # # image only
    # interpreter.allocate_tensors()
    # input_details = interpreter.get_input_details()[0]
    # output_details = interpreter.get_output_details()[0]
    # if input_details['dtype'] == np.uint8 or input_details['dtype'] == np.float16:
    #     input_scale, input_zero_point = input_details["quantization"]
    #     # print(input_scale)
    #     # print(input_zero_point)
    #     test_images = test_images / input_scale + input_zero_point
    # test_images = test_images.astype(input_details["dtype"])
    # split_len = 50
    # total_predictions = np.array([])
    # for i in range(200):
    #     print(i)
    #     # if i % 20 == 0:
    #     #     print("batch: ", i)
    #     batch_image = test_images[int(i * split_len): int((i + 1) * split_len)]
    #     interpreter.set_tensor(input_details["index"], batch_image)
    #     interpreter.invoke()
    #     output = interpreter.get_tensor(output_details["index"])
    #     predictions = output.argmax(axis=1)
    #     print(predictions)
    #     total_predictions = np.concatenate((total_predictions, predictions))
    # quan_acc = (np.sum(labels == total_predictions) * 100) / len(labels)
    print("data : {}, acc : {}".format(data_path, quan_acc))
    csv_file = open(save_path, "a")
    try:
        writer = csv.writer(csv_file)
        writer.writerow([model_path, data_path, quan_acc])
    finally:
        csv_file.close()
    np.save(label_save_path, total_predictions)


def mnist_imagenet_run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", "-data_type",
                        type=str,
                        default='mnist'
                        )
    parser.add_argument("--model_type", "-model_type",
                        type=str,
                        default='lenet1'
                        )
    parser.add_argument("--data_path", "-data_path",
                        type=str,
                        default='brightness'
                        )
    parser.add_argument("--model_path", "-model_path",
                        type=str,
                        default='../models/imagenet/tflite/lenet1_acc.h5'
                        )
    parser.add_argument("--save_path", "-save_path",
                        type=str,
                        default='../results/RQ1/lenet1_acc.csv'
                        )

    args = parser.parse_args()
    data_type = args.data_type
    model_type = args.model_type
    save_path = args.save_path
    data_path = args.data_path
    model_path = args.model_path
    if data_type == 'mnist' or data_type == 'imdb':
        tflite_val_main(data_type, model_type, save_path)
    else:
        tflite_val_main_imagenet(data_type, data_path, model_path, save_path)


def cifar10_run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-data_path",
                        type=str,
                        default='brightness',
                        )
    parser.add_argument("--model_path", "-model_path",
                        type=str,
                        default='../models/cifar10/tflite/nin-16bits-quan.lite'
                        )
    parser.add_argument("--save_path", "-save_path",
                        type=str,
                        default='../results/RQ1/tflite_cifar10_nin_acc.csv'
                        )
    parser.add_argument("--label_save_path", "-label_save_path",
                        type=str,
                        default='../results/RQ3/cifar10/tflite_nin_16bits_brightness.npy'
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
    tflite_val_main_cifar10(data_path, model_path, save_path, label_save_path, severity)


if __name__ == '__main__':
    # cifar10_run()
    tflite_val_main_ori("imdb", "../results/RQ4/imdb/tflite_imdb_ori_acc.csv")
    # model_paths = [
    #     "../models/iwildscam/tflite/densenet-8bit-quan.lite",
    #     # "../models/iwildscam/tflite/resnet50_adv2_8bits.lite"
    # ]
    # for model_path in model_paths:
    #     tflite_val_main_imagenet("wilds", "no", model_path, "../results/RQ1/tflite_wilds_ori_acc.csv")

# python tflite_evaluation.py -data_type imdb -model_type lstm -save_path ../results/RQ1/tflite_lstm_acc.h5
# python tflite_evaluation.py -data_type mnist -model_type lenet1 -save_path ../results/RQ1/tflite_mnist_lenet1_acc.csv

# python tflite_evaluation.py -data_type imagenet -data_path brightness -model_path ../models/imagenet/tflite/resnet101-16bits-quan.lite -save_path ../results/RQ1/tflite_imagenet_resnet_acc.h5
# python tflite_evaluation.py -data_type imagenet -data_path brightness -model_path ../models/imagenet/tflite/resnet101-8bits-quan.lite -save_path ../results/RQ1/tflite_imagenet_resnet_acc.csv
# python tflite_evaluation.py -data_type imagenet -data_path contrast -model_path ../models/imagenet/tflite/resnet101_mixup-8bits-quan.lite -save_path ../results/RQ1/tflite_imagenet_resnet_acc.csv
