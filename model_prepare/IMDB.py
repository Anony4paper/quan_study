import tensorflow as tf
import numpy as np
import re
import collections
import sklearn.metrics as sk
# import keras
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Newsgroups import *
from tensorflow_model_optimization.python.core.quantization.keras import quantize_config
import tensorflow_model_optimization as tfmot


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
            y.append(line[-1])
    return x, np.array(y, dtype=int)


def get_vocab(dataset):
    '''
    :param dataset: the text from load_data

    :return: a _ordered_ dictionary from words to counts
    '''
    vocab = {}

    # create a counter for each word
    for example in dataset:
        example_as_list = example.split()
        for word in example_as_list:
            vocab[word] = 0

    for example in dataset:
        example_as_list = example.split()
        for word in example_as_list:
            vocab[word] += 1

    # sort from greatest to least by count
    return collections.OrderedDict(sorted(vocab.items(), key=lambda x: x[1], reverse=True))


def text_to_rank(dataset, _vocab, desired_vocab_size=5000):
    '''
    :param dataset: the text from load_data
    :vocab: a _ordered_ dictionary of vocab words and counts from get_vocab
    :param desired_vocab_size: the desired vocabulary size
    words no longer in vocab become UUUNNNKKK
    :return: the text corpus with words mapped to their vocab rank,
    with all sufficiently infrequent words mapped to UUUNNNKKK; UUUNNNKKK has rank desired_vocab_size
    (the infrequent word cutoff is determined by desired_vocab size)
    '''
    _dataset = dataset[:]  # aliasing safeguard
    vocab_ordered = list(_vocab)
    count_cutoff = _vocab[vocab_ordered[desired_vocab_size - 1]]  # get word by its rank and map to its count

    word_to_rank = {}
    for i in range(len(vocab_ordered)):
        # we add one to make room for any future padding symbol with value 0
        word_to_rank[vocab_ordered[i]] = i + 1

    # we need to ensure that other words below the word on the edge of our desired_vocab size
    # are not also on the count cutoff, so we subtract a bit
    # this is likely quicker than adding another preventative if case
    for i in range(50):
        _vocab[vocab_ordered[desired_vocab_size + i]] -= 0.1

    for i in range(len(_dataset)):
        example = _dataset[i]
        example_as_list = example.split()
        for j in range(len(example_as_list)):
            try:
                if _vocab[example_as_list[j]] >= count_cutoff:
                    example_as_list[j] = word_to_rank[example_as_list[j]]
                else:
                    example_as_list[j] = desired_vocab_size  # UUUNNNKKK
            except:
                example_as_list[j] = desired_vocab_size  # UUUNNNKKK
        _dataset[i] = example_as_list

    return _dataset


###   Clone NoOpQuantizeConfig locally   ###
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


def IMDB_LSTM():
    max_features = 20000  # Only consider the top 20k words
    inputs = keras.Input(shape=(None,), dtype="int32")
    x = layers.Embedding(max_features, 128)(inputs)
    x = layers.SpatialDropout1D(0.3)(x)
    x = layers.LSTM(50, recurrent_dropout=0.3)(x)
    x = layers.Dropout(0.3)(x)
    # x = layers.Dense(20, activation="relu")(x)
    outputs = layers.Dense(2, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model


def IMDB_LSTM_QA():
    max_features = 20000  # Only consider the top 20k words
    # inputs = tfmot.quantization.keras.quantize_annotate_layer(layers.InputLayer(input_shape=(None,), dtype="int32"), quantize_config=quantize_config)
    # inputs = layers.InputLayer(input_shape=(None,), dtype="int32")
    # x = layers.Embedding(max_features, 128)(inputs)
    # x = layers.SpatialDropout1D(0.3)(x)
    # x = layers.LSTM(50, recurrent_dropout=0.3)(x)
    # x = layers.Dropout(0.3)(x)
    # # x = layers.Dense(20, activation="relu")(x)
    # outputs = layers.Dense(2, activation="sigmoid")(x)
    # model = keras.Model(inputs, outputs)
    # model.summary()

    model = keras.Sequential(
        [
            tfmot.quantization.keras.quantize_annotate_layer(layers.InputLayer(input_shape=(None,), dtype="int32"),
                                                             quantize_config=quantize_config),
            tfmot.quantization.keras.quantize_annotate_layer(layers.Embedding(max_features, 128), quantize_config=quantize_config),
            tfmot.quantization.keras.quantize_annotate_layer(layers.SpatialDropout1D(0.3), quantize_config=quantize_config),
            tfmot.quantization.keras.quantize_annotate_layer(layers.LSTM(50, recurrent_dropout=0.3), quantize_config=quantize_config),
            tfmot.quantization.keras.quantize_annotate_layer(layers.Dropout(0.3), quantize_config=quantize_config),
            layers.Dense(2),
            tfmot.quantization.keras.quantize_annotate_layer(layers.Activation("sigmoid"), quantize_config=quantize_config)
        ]
    )
    model.summary()
    return model


def IMDB_GRU():
    max_features = 20000  # Only consider the top 20k words
    inputs = keras.Input(shape=(None,), dtype="int32")
    x = layers.Embedding(max_features, 128)(inputs)
    x = layers.SpatialDropout1D(0.3)(x)
    x = layers.GRU(75, recurrent_dropout=0.3)(x)
    x = layers.Dropout(0.3)(x)
    # x = layers.Dense(20, activation="relu")(x)
    outputs = layers.Dense(2, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model


def IMDB_GRU_QA():
    max_features = 20000  # Only consider the top 20k words
    model = keras.Sequential(
        [
            tfmot.quantization.keras.quantize_annotate_layer(layers.InputLayer(input_shape=(None,), dtype="int32"),
                                                             quantize_config=quantize_config),
            # tfmot.quantization.keras.quantize_annotate_layer(layers.Embedding(max_features, 128),
            #                                                  quantize_config=quantize_config),
            # tfmot.quantization.keras.quantize_annotate_layer(layers.SpatialDropout1D(0.3),
            #                                                  quantize_config=quantize_config),
            # tfmot.quantization.keras.quantize_annotate_layer(layers.GRU(50, recurrent_dropout=0.3),
            #                                                  quantize_config=quantize_config),
            # tfmot.quantization.keras.quantize_annotate_layer(layers.Dropout(0.3), quantize_config=quantize_config),
            layers.Embedding(max_features, 128),
            layers.SpatialDropout1D(0.3),
            layers.GRU(50, recurrent_dropout=0.3),
            layers.Dropout(0.3),
            layers.Dense(2),
            tfmot.quantization.keras.quantize_annotate_layer(layers.Activation("sigmoid"),
                                                             quantize_config=quantize_config)
        ]
    )
    # inputs = tfmot.quantization.keras.quantize_annotate_layer(keras.Input(shape=(None,), dtype="int32"), quantize_config=quantize_config)
    # x = layers.Embedding(max_features, 128)(inputs)
    # x = layers.SpatialDropout1D(0.3)(x)
    # x = layers.GRU(75, recurrent_dropout=0.3)(x)
    # x = layers.Dropout(0.3)(x)
    # # x = layers.Dense(20, activation="relu")(x)
    # outputs = layers.Dense(2, activation="sigmoid")(x)
    # model = keras.Model(inputs, outputs)
    # model.summary()
    return model


def preprocessing_text(texts):
    texts = re.sub(r'<.*?>', '', texts)
    texts = re.sub(r'[^a-zA-Z]', ' ', texts)
    return ' '.join(x.lower() for x in texts.split())


def model_eval(model_path):
    max_example_len = 200
    model = keras.models.load_model(model_path)
    model.summary()
    df = pd.read_csv("data/IMDB/IMDB_Dataset.csv")
    df['review_cleaned'] = df['review'].apply(lambda x: preprocessing_text(x))
    y = df['sentiment'].map({'negative': 0, 'positive': 1}).values
    # print(df.head())
    x_train = df['review_cleaned'][:25000]
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(x_train)
    x_test = df['review_cleaned'][25000: 35000]
    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=max_example_len)
    y_test = y[25000: 35000]
    y_test = to_categorical(y_test, 2)
    score = model.evaluate(x_test, y_test)
    print("loss: ", score[0])
    print("acc: ", score[1])


if __name__ == '__main__':
    ########### Normal training ################
    # max_example_len = 200
    # batch_size = 32
    # embedding_dims = 50
    # vocab_size = 19999
    # num_epochs = 15
    #
    # df = pd.read_csv("data/IMDB/IMDB_Dataset.csv")
    # max_example_len = 200
    # df['review_cleaned'] = df['review'].apply(lambda x: preprocessing_text(x))
    # # print(df.head())
    #
    # x_train = df['review_cleaned'][:25000]
    #
    # tokenizer = Tokenizer(num_words=20000)
    # tokenizer.fit_on_texts(x_train)
    # x_train = tokenizer.texts_to_sequences(x_train)
    # x_train = pad_sequences(x_train, maxlen=max_example_len)
    # y = df['sentiment'].map({'negative': 0, 'positive': 1}).values
    # y_train = y[:25000]
    # y_train = to_categorical(y_train, 2)
    #
    # x_test = df['review_cleaned'][25000: 35000]
    # x_test = tokenizer.texts_to_sequences(x_test)
    # x_test = pad_sequences(x_test, maxlen=max_example_len)
    # y_test = y[25000: 35000]
    # y_test = to_categorical(y_test, 2)
    #
    # # model = IMDB_LSTM()
    # model = IMDB_GRU()
    # model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    # model.summary()
    # # his = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))
    # # model.save("models/IMDB_gru.h5")

    ################ OE training ################
    # train_IMDB_yelp_oe("data/IMDB/yelp_OOD_data.npy")

    ###############  Evaluation #####################
    model_path = "models/IMDB/IMDB_gru.h5"
    model_eval(model_path)

# CUDA_VISIBLE_DEVICES=2 python IMDB.py
