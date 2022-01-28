import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Activation, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


def ResNet50_model():
    base_model = tf.keras.applications.ResNet50V2(
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
    model = ResNet50_model()
    model.summary()
    # configures = model.get_config()
    # print(configures)
    img_path = '../../tint-imagenet/tiny-imagenet-200/test/images/test_0.JPEG'
    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print(np.argmax(preds, axis=1))
    # print('Predicted:', decode_predictions(preds, top=3)[0])

