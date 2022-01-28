import tensorflow as tf
import json
import numpy as np
import sys
sys.path.append('../model_prepare')
from model_training import *


def dis_retrain(json_file, data_type, model_type, save_path):
    with open(json_file) as json_file:
        data = json.load(json_file)

    if data_type == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        if model_type == 'lenet1':
            model = tf.keras.models.load_model("../models/mnist/lenet1.h5")
            selected_index = data['results'][0]["union_index"]
            x_selected = x_test[selected_index]
            y_selected = y_test[selected_index]
            x_combine = np.concatenate((x_train, x_selected))
            y_combine = np.concatenate((y_train, y_selected))
            model.fit(x_combine,
                      y_combine,
                      batch_size=256,
                      shuffle=True,
                      epochs=5,
                      validation_data=(x_test, y_test),
                      verbose=1
                      )
            model.save(save_path)
        elif model_type == 'lenet5':
            model = tf.keras.models.load_model("../models/mnist/lenet5.h5")
            selected_index = data['results'][0]["union_index"]
            x_selected = x_test[selected_index]
            y_selected = y_test[selected_index]
            x_combine = np.concatenate((x_train, x_selected))
            y_combine = np.concatenate((y_train, y_selected))
            model.fit(x_combine,
                      y_combine,
                      batch_size=256,
                      shuffle=True,
                      epochs=5,
                      validation_data=(x_test, y_test),
                      verbose=1
                      )
            model.save(save_path)

    elif data_type == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        if model_type == 'nin':
            model = tf.keras.models.load_model("../models/cifar10/nin.h5")
            selected_index = data['results'][0]["union_index"]
            x_selected = x_test[selected_index]
            y_selected = y_test[selected_index]
            x_combine = np.concatenate((x_train, x_selected))
            y_combine = np.concatenate((y_train, y_selected))
            checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path,
                                                            monitor='val_accuracy',
                                                            verbose=1,
                                                            save_best_only=True,
                                                            mode='max',
                                                            period=1)
            model.fit(x_combine,
                      y_combine,
                      batch_size=128,
                      epochs=10,
                      validation_data=(x_test, y_test),
                      verbose=1,
                      callbacks=[checkpoint]
                      )
        else:
            model = tf.keras.models.load_model("../models/cifar10/resnet20.h5")
            selected_index = data['results'][0]["union_index"]
            x_selected = x_test[selected_index]
            y_selected = y_test[selected_index]
            x_combine = np.concatenate((x_train, x_selected))
            y_combine = np.concatenate((y_train, y_selected))
            checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path,
                                                            monitor='val_accuracy',
                                                            verbose=1,
                                                            save_best_only=True,
                                                            mode='max',
                                                            period=1)
            model.fit(x_combine,
                      y_combine,
                      batch_size=128,
                      epochs=10,
                      validation_data=(x_test, y_test),
                      verbose=1,
                      callbacks=[checkpoint]
                      )

    elif data_type == 'imdb':
        max_example_len = 200
        batch_size = 32
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
        y_test = to_categorical(y_test, 2)
        if model_type == 'lstm':
            model = tf.keras.models.load_model("../models/imdb/lstm_5000.h5")
        else:
            model = tf.keras.models.load_model("../models/imdb/gru_5000.h5")
        selected_index = data['results'][0]["union_index"]
        x_selected = x_test[selected_index]
        y_selected = y_test[selected_index]
        x_combine = np.concatenate((x_train, x_selected))
        y_combine = np.concatenate((y_train, y_selected))
        checkPoint = tf.keras.callbacks.ModelCheckpoint(save_path,
                                                        monitor="val_accuracy",
                                                        save_best_only=True,
                                                        verbose=0)
        model.fit(x_combine,
                  y_combine,
                  epochs=5,
                  batch_size=batch_size,
                  validation_data=(x_test, y_test),
                  callbacks=[checkPoint])
    else:
        dataset = get_dataset(dataset='iwildcam', download=False, root_dir="../model_evaluation/data")
        train_data = dataset.get_subset('train',
                                        transform=transforms.Compose([transforms.Resize((448, 448)),
                                                                      transforms.ToTensor()]))
        val_data = dataset.get_subset('val',
                                      transform=transforms.Compose([transforms.Resize((448, 448)),
                                                                    transforms.ToTensor()]))
        test_data = dataset.get_subset('id_test',
                                       transform=transforms.Compose([transforms.Resize((448, 448)),
                                                                     transforms.ToTensor()]))
        train_loader = get_train_loader('standard', train_data, batch_size=16)
        val_loader = get_train_loader('standard', val_data, batch_size=16)
        test_loader = get_eval_loader('standard', test_data, batch_size=1)
        train_loader = DataGenerator(train_loader, 182)
        val_loader = DataGenerator(val_loader, 182)
        test_loader = DataGenerator(test_loader, 182)
        if model_type == 'resnet50':
            model = tf.keras.models.load_model("../models/iwildscam/resnet50.h5")
            base_acc = 0.7601
        else:
            model = tf.keras.models.load_model("../models/iwildscam/densenet.h5")
            base_acc = 0.7578
        selected_index = data['results'][0]["union_index"]
        x_test = []
        y_test = []
        flag = 0
        for x, y in test_loader:
            if flag in selected_index:
                x_test.append(x)
                y_test.append(y)
            flag += 1
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)
        epoch_num = 5
        steps_per_epoch = 129809 // 16
        valid_per_step = 7314 // 16

        x_test = x_test.reshape(x_test.shape[0], x_test.shape[2], x_test.shape[3], x_test.shape[4])
        y_test = y_test.reshape(y_test.shape[0], y_test.shape[2])
        print(x_test.shape)
        print(y_test.shape)
        for _ in range(epoch_num):
            model.fit_generator(train_loader,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=val_loader,
                                validation_steps=valid_per_step,
                                epochs=1)
            model.fit(x_test,
                      y_test,
                      epochs=1)
            _, acc = model.evaluate_generator(val_loader,
                                              valid_per_step,
                                              use_multiprocessing=True,
                                              workers=12)
        model.save(save_path)


if __name__ == "__main__":
    # MNIST
    # json_file = "../results/RQ4/mnist/lenet5.json"
    # data_type = "mnist"
    # model_type = "lenet5"
    # save_path = "../models/RQ4/mnist/lenet5.h5"
    # dis_retrain(json_file, data_type, model_type, save_path)

    # cifar10
    # json_file = "../results/RQ4/cifar10/resnet20.json"
    # data_type = "cifar10"
    # model_type = "resnet20"
    # save_path = "../models/RQ4/cifar10/resnet20.h5"
    # dis_retrain(json_file, data_type, model_type, save_path)

    # wilds
    json_file = "../results/RQ4/wilds/resnet50.json"
    data_type = "wilds"
    model_type = "resnet50"
    save_path = "../models/RQ4/wilds/resnet50.h5"
    dis_retrain(json_file, data_type, model_type, save_path)

    json_file = "../results/RQ4/wilds/densenet.json"
    data_type = "wilds"
    model_type = "densenet"
    save_path = "../models/RQ4/wilds/densenet.h5"
    dis_retrain(json_file, data_type, model_type, save_path)
    # imdb
    # json_file = "../results/RQ4/imdb/gru.json"
    # data_type = "imdb"
    # model_type = "gru"
    # save_path = "../models/RQ4/imdb/gru.h5"
    # dis_retrain(json_file, data_type, model_type, save_path)

#

