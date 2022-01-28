import pickle
import glob
import numpy as np
from keras.models import load_model
from keras.datasets import mnist
import keras
#
# with open('results/mnist-tflite/lenet1_quan_0_a/000000.pickle', 'rb') as f:
#     data = pickle.load(f)
#     print(data.mutant.shape)
#     print(data.seed_label)


def read_mutants(folder):
    mutants = []
    labels = []
    paths = glob.glob(folder + "*.npy")
    for file_path in paths:
        # print(file_path)
        data = np.load(file_path)
        # print(file_path.split('/')[-1].split('_')[0])
        label = int(file_path.split('/')[-1].split('_')[0])
        mutants.append(data)
        labels.append(label)
    mutants = np.array(mutants)
    labels = np.asarray(labels)
    print(mutants.shape)
    # print(labels.shape)
    return mutants, labels


def retrain_model(ori_model_path, mutants_folder, model_save_path):
    original_model = load_model(ori_model_path)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    x_mutants, y_mutants = read_mutants(mutants_folder)
    # print(x_mutants.shape)
    x_mutants = x_mutants.astype('float32') / 255
    # print(x_mutants[0])

    y_test = keras.utils.to_categorical(y_test, 10)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_mutants = keras.utils.to_categorical(y_mutants, 10)
    x_train_total = np.concatenate((x_train, x_mutants))
    y_train_total = np.concatenate((y_train, y_mutants))

    original_model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
    original_model.summary()
    his = original_model.fit(x_train_total,
                             y_train_total,
                             batch_size=256,
                             shuffle=True,
                             epochs=20,
                             validation_data=(x_test, y_test),
                             verbose=1)
    original_model.save(model_save_path)


if __name__ == '__main__':
    original_model_path = "models/lenet1.h5"
    mutants_folder = "ori_output/crashes/"
    model_save_path = "models/lenet1_retrained.h5"
    retrain_model(original_model_path, mutants_folder, model_save_path)
    # read_mutants(mutants_folder)

