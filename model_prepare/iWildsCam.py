import tensorflow as tf
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from tensorflow.keras import optimizers
import numpy as np
import tensorflow.keras as k
import torchvision.transforms as transforms
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Activation, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob
from PIL import Image


class DataGenerator(k.utils.Sequence):
    """
        class to be fed into model.fit_generator method of tf.keras model
        uses a pytorch dataloader object to create a new generator object that can be used by tf.keras
        dataloader in pytorch must be used to load image data
        transforms on the input image data can be done with pytorch, model fitting still with tf.keras
        ...
        Attributes
        ----------
        gen : torch.utils.data.dataloader.DataLoader
            pytorch dataloader object; should be able to load image data for pytorch model
        ncl : int
            number of classes of input data; equal to number of outputs of model
    """
    def __init__(self, gen, ncl):
        """
            Parameters
            ----------
            gen : torch.utils.data.dataloader.DataLoader
                pytorch dataloader object; should be able to load image data for pytorch model
            ncl : int
                number of classes of input data; equal to number of outputs of model
        """
        self.gen = gen
        self.iter = iter(gen)
        self.ncl = ncl

    def __getitem__(self, _):
        """
            function used by model.fit_generator to get next input image batch
            Variables
            ---------
            ims : np.ndarray
                image inputs; tensor of (batch_size, height, width, channels); input of model
            lbs : np.ndarray
                labels; tensor of (batch_size, number_of_classes); correct outputs for model
        """
        # catch when no items left in iterator
        try:
            ims, lbs, _ = next(self.iter)  # generation of data handled by pytorch dataloader
        # catch when no items left in iterator
        except StopIteration:
            self.iter = iter(self.gen)  # reinstanciate iteator of data
            ims, lbs, _ = next(self.iter)  # generation of data handled by pytorch dataloader
        # swap dimensions of image data to match tf.keras dimension ordering
        ims = np.swapaxes(np.swapaxes(ims.numpy(), 1, 3), 1, 2)
        # convert labels to one hot representation
        lbs = np.eye(self.ncl)[lbs]
        # print(ims.shape)
        # print(lbs.shape)
        lbs = lbs.reshape(-1, self.ncl)
        # print(lbs.shape)
        return ims, lbs

    def __len__(self):
        """
            function that returns the number of batches in one epoch
        """
        return len(self.gen)


def image_read(file_folder):
    file_list = glob.glob(file_folder)
    transformer = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()])
    x_list = []
    y_list = []
    count_num = 0
    for file_path in file_list:
        count_num += 1
        if count_num % 1000 == 0:
            print(count_num)
        img = Image.open(file_path)
        # img_np = np.asarray(img)
        # x_list.append(transforms.ToTensor()(img).numpy())
        x_list.append(transformer(img).numpy())
        y_list.append(int(file_path.split('/')[-1].split('_')[-1].split('.')[0]))
        if count_num > 10000:
            break
        # img_np = transformer(img_np)
        # print(img_np.shape)

    x_list = np.asarray(x_list)
    y_list = np.asarray(y_list)
    x_list = np.swapaxes(np.swapaxes(x_list, 1, 3), 1, 2)
    # x_list = transformer(x_list)
    # y_list = to_categorical(y_list, 182)
    # print(x_list.shape)
    # print(y_list.shape)
    return x_list, y_list


def iwildscam_model_ResNet50():
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        # weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=182,
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Reshape((-1, 2))(x)
    x = Dense(182)(x)
    predictions = Activation('softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def iwildscam_model_DenseNet():
    base_model = tf.keras.applications.DenseNet121(
        include_top=False,
        # weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=182,
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Reshape((-1, 2))(x)
    x = Dense(182)(x)
    predictions = Activation('softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def train_normal():
    # base_model = tf.keras.applications.ResNet50(
    #     include_top=False,
    #     # weights="imagenet",
    #     input_tensor=None,
    #     input_shape=None,
    #     pooling=None,
    #     classes=182,
    # )
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        # weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=182,
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Reshape((-1, 2))(x)
    x = Dense(182)(x)

    predictions = Activation('softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=3e-5),
                  metrics=['accuracy'])
    model.summary()

    print(model.output_shape)
    dataset = get_dataset(dataset='iwildcam', download=False)

    train_data = dataset.get_subset('train',
                                    transform=transforms.Compose([transforms.Resize((448, 448)),
                                                                 transforms.ToTensor()]))
    val_data = dataset.get_subset('val',
                                  transform=transforms.Compose([transforms.Resize((448, 448)),
                                                                transforms.ToTensor()]))

    # print(dir(train_data))
    train_loader = get_train_loader('standard', train_data, batch_size=16)
    val_loader = get_train_loader('standard', val_data, batch_size=16)
    dataloader = DataGenerator(train_loader, 182)
    val_dataloader = DataGenerator(val_loader, 182)

    steps_per_epoch = 129809 // 16

    # for d, l in dataloader:
    #     print(l[0])
    #     break
    checkPoint = tf.keras.callbacks.ModelCheckpoint("models/iwildcam/ResNet50_half_best.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
    model.fit_generator(dataloader,
                        steps_per_epoch=100,
                        validation_data=val_dataloader,
                        # validation_steps=10,
                        epochs=5,
                        callbacks=[checkPoint])
    model.save("models/iwildcam/ResNet50_half.h5")


if __name__ == "__main__":
    # save_path = "models/iwildcam/ResNet50_oe.h5"
    # train_or_wilds(save_path)
    train_normal()
