from face_recognition import face_locations
from keras import Model
from keras.applications.xception import Xception
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, GlobalAveragePooling2D

import ml_utils


class FaceEmotion:
    """Class for recognizing emotion using default Deep Learning Xception model"""
    def __init__(self, input_shape=(200, 200, 3)):  # TODO: Check input_shape
        """Initialize main parameters of FaceEmotion class
        :param input_shape: Input images shape
        """
        self.input_shape = input_shape

        self.model = Xception(include_top=False, input_shape=input_shape)
        self.model = self.add_classificator(self.model)

    @staticmethod
    def add_classificator(base_model):
        """Add a classificator to a model
        :param base_model: Keras model object
        """
        layer = base_model.output
        layer = GlobalAveragePooling2D(name="classificator_block_pool")(layer)
        layer = Dense(512, activation='relu', name='classificator_block_dense_1')(layer)
        layer = Dense(64, activation='relu', name='classificator_block_dense_2')(layer)
        layer = Dense(6, activation='relu', name='classificator_block_dense_3')(layer)

        model = Model(inputs=base_model.input, outputs=layer)

        # freeze early layers
        for l in base_model.layers:
            l.trainable = False

        model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

        return model

    def model_architecture(self, filename=None):
        """Show model architecture and save it to file
        :param filename: Path to the model architecture image file
        """
        list_summary = []
        self.model.summary(print_fn=lambda x: list_summary.append(x))
        summary = "\n".join(list_summary)

        if filename:
            with open(filename + '.txt', 'w') as f:
                f.write(summary)

            from keras.utils import plot_model
            plot_model(self.model, filename + '.jpg')

        return summary

    # noinspection PyShadowingNames
    def train(self, generator, epochs, steps_per_epoch):
        """Train model
        :param generator: Data generator compatible with Keras model
        :param epochs: Number of epochs to train model
        :param steps_per_epoch: Number of faces used in one step
        """
        stopper = EarlyStopping(patience=100)  # , restore_best_weights=True)
        save_dir = "training/e{epoch:02d}-a{acc:.2f}.ckpt"
        saver = ModelCheckpoint(save_dir)  # , save_best_only=True)

        self.model.fit_generator(generator, steps_per_epoch, epochs, callbacks=[stopper, saver])

    def get_emotion(self, image):
        emotions = []
        for top, right, bottom, left in face_locations(image):
            emotion = self.model.predict(image[top:bottom, left:right])
            emotions.append(emotion)
        return emotions


if __name__ == "__main__":
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    face = FaceEmotion()

    dataset_dir = "D:\\ML\\datasets\\face_emotions"

    epochs = 100
    batch_size = 500
    steps_per_epoch = ml_utils.get_steps_per_epoch(dataset_dir, batch_size)

    generator = ml_utils.generator(dataset_dir, batch_size)
    face.train(generator, epochs, steps_per_epoch)
