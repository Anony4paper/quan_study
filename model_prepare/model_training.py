from IMDB import *
from iWildsCam import *
from LeNet import *
from keras.datasets import mnist
from wilds import get_dataset
import argparse
import tensorflow_addons as tfa
from ResNet20 import *
from NiN import *
from tensorflow import keras

def schedule(epoch_idx):
    if (epoch_idx + 1) < 10:
        return 1e-03
    elif (epoch_idx + 1) < 20:
        return 1e-04  # lr_decay_ratio = 0.2
    elif (epoch_idx + 1) < 30:
        return 1e-05
    return 1e-05


def training(data_type, model_type, save_path):
    if data_type == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        y_test = keras.utils.to_categorical(y_test, 10)
        y_train = keras.utils.to_categorical(y_train, 10)

        batch_size = 256
        epochs = 20
        if model_type == 'lenet1':
            model = LeNet1_model()
        else:
            model = LeNet5_model()
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        his = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        shuffle=True,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        verbose=1)

        print(his.history['val_accuracy'])
        model.save(save_path)

    elif data_type == 'cifar10':
        if model_type == 'resnet20':
            train_resnet()
        else:
            train_nin()

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
        y_test = to_categorical(y_test, 2)
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
        model.fit(x_train,
                  y_train,
                  epochs=num_epochs,
                  batch_size=batch_size,
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

        model.fit_generator(dataloader,
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
    training(data_type, model_type, save_path)

# CUDA_VISIBLE_DEVICES=1 python model_training.py -data imagenet -model resnet101 -save ../models/imagenet/resnet101.h5
# CUDA_VISIBLE_DEVICES=0 python model_training.py -data imagenet -model densenet -save ../models/imagenet/Densenet.h5
# CUDA_VISIBLE_DEVICES=1 python model_training.py -data iwildscam -model densenet -save ../models/iwildscam/DenseNet.h5
# python model_training.py -data imdb -model lstm -save ../models/imdb/lstm_test.h5
