import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Activation, Flatten, Reshape
from tensorflow.keras.models import Model


def DenseNet_model():
    base_model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=200
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(200)(x)
    predictions = Activation('softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


if __name__ == "__main__":
    model = DenseNet_model()
    model.summary()
