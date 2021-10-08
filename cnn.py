import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import *

from sklearn.metrics import classification_report, confusion_matrix
# to compute weights for the class distribution
from sklearn.utils.class_weight import compute_class_weight
import wandb
from wandb.keras import WandbCallback

import opt


class Cnn:
    def __init__(self, IMAGE_SHAPE, RGB, MODE, NO_PLOT, SINGLE_RUN_CONFIG, class_distribution=None):
        """
        Create a model

        Parameters:
        -----------
        IMAGE_SHAPE: tuple
            The shape of input images in the format (width, height)
        RGB: bool
            Whether the input images have three or one image channels.
        MODE: string
            Whether a new model should or a saved model is used when using cnn.load
        SINGLE_RUN_HYPERPARAMETER: dict
            dict with model parameters
        class_distribution: string, optional
            Gives the class distribution to rescale loss in case of unbalanced classes
        Returns:
        --------
        model: keras Model
            The created model.
        """
        if not SINGLE_RUN_CONFIG.exclude_wandb:
            wandb.login()
            wandb.init(project='covid19', entity='cov01', config=SINGLE_RUN_CONFIG)
            self.config = wandb.config
        else:
            self.config = SINGLE_RUN_CONFIG

        self.class_distribution = class_distribution
        self.IMAGE_SHAPE = IMAGE_SHAPE
        self.RGB = RGB
        self.MODE = MODE
        self.NO_PLOT = NO_PLOT

        if not self.config.use_pretrained:
            layer_in = Input(shape=(*IMAGE_SHAPE, 3 if RGB else 1), name="input")
            for i in range(self.config.conv_sections):
                filters = self.config.filter * self.config.filter_scaling ** i
                window = (self.config[f"conv_window_{i}"], self.config[f"conv_window_{i}"])
                upsampling = (int(self.config.upsampling), int(self.config.upsampling))

                for j in range(self.config.conv_layers_per_section - 1):
                    x = Conv2D(filters,
                               window,
                               activation=self.config.activation_conv,
                               kernel_initializer="HeNormal",
                               padding="same", name=f"conv{i}-{j}")(layer_in if i == 0 and j == 0 else x)
                    x = BatchNormalization()(x) if self.config.batchnorm else x

                if upsampling[0] > 1 and i == self.config.conv_layers_per_section - 1:
                    x = UpSampling2D(size=upsampling, data_format="channels_last", interpolation="nearest")(x)

                x = Conv2D(filters,
                           window,
                           activation=self.config.activation_conv,
                           kernel_initializer="HeNormal",
                           padding="same",
                           name=(
                               "activation" if i == self.config.conv_layers_per_section - 1 else f"conv{i}-{self.config.conv_layers_per_section - 1}"))(
                    layer_in if i == 0 and self.config.conv_layers_per_section == 1 else x)
                x = BatchNormalization()(x) if self.config.batchnorm else x

                if self.config.pooling and i < self.config.conv_sections - 1:
                    x = MaxPooling2D((self.config.pool_window, self.config.pool_window))(x)

            if self.config.gap:
                x = GlobalAveragePooling2D()(x)

            elif self.config.pooling:
                x = MaxPooling2D((self.config.pool_window, self.config.pool_window))(x)

            x = Flatten()(x)

            for i in range(self.config.dense_layers):
                x = Dense(int(self.config.dense_neurons * self.config.dense_scaling ** i),
                          activation=self.config.activation_dense,
                          kernel_initializer="HeNormal")(x)
                x = Dropout(self.config.dropout)(x)

            layer_out = Dense(4, activation=self.config.activation_output, dtype='float32', name="output")(x)

            self.model = Model(inputs=layer_in, outputs=layer_out)

        else:
            self.model = self.get_pretrained()

        self.model.compile(optimizer=self.get_optimizer(self.config.optimizer, self.config.learning_rate),
                           loss=self.config.loss,
                           metrics=["accuracy"])
        self.model.summary()
        self.history = None

    def train(self):
        """
        Trains the model, applies data augmentation when loading data from directory
        """
        if not self.config.exclude_wandb:
            wandb_callback = wandb.keras.WandbCallback(log_weights=True)

        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            # min_delta=1e-3,
            patience=self.config.patience,
            restore_best_weights=True,
            verbose=1
        )

        train = ImageDataGenerator(
            rescale=self.config.rescale,
            shear_range=45,
            zoom_range=[0.6, 1.2],
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.2, 1.5],
            rotation_range=20,
            fill_mode="nearest",
            zca_whitening=self.config.zca_whitening
        ).flow_from_directory(
            f"{self.config.data_name}/train",
            target_size=self.IMAGE_SHAPE,
            batch_size=self.config.batch_size,
            shuffle=True,
            color_mode="rgb" if self.RGB else "grayscale"
        )

        test = ImageDataGenerator(
            rescale=self.config.rescale,
            zca_whitening=self.config.zca_whitening
        ).flow_from_directory(
            f"{self.config.data_name}/test",
            target_size=self.IMAGE_SHAPE,
            batch_size=self.config.batch_size,
            shuffle=True,
            color_mode="rgb" if self.RGB else "grayscale"
        )
        self.history = self.model.fit(
            train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=test,
            callbacks=[wandb_callback, earlystop_callback]
        ).history

    def validate(self):
        """
        validates the model and prints validation accuracy and loss as well as the confusion matrix
        """
        validation = ImageDataGenerator(
            rescale=self.config.rescale,
            zca_whitening=self.config.zca_whitening
        ).flow_from_directory(
            f"{self.config.data_name}/validation",
            target_size=self.IMAGE_SHAPE,
            batch_size=self.config.batch_size,
            color_mode="rgb" if self.RGB else "grayscale",
            shuffle=False
        )
        loss, acc = self.model.evaluate(validation)
        print(f"validation: loss={loss} acc={acc}")

        pred = np.argmax(self.model.predict(validation), axis=1)

        print("confusion matrix")
        print(confusion_matrix(validation.classes, pred))

        classes = ["atypical", "indeterminate", "negative", "typical"]
        print(classification_report(validation.classes, pred, target_names=classes))

    def plot(self, path=None):
        """
        Plot the training self.history.

        Parameters:
        -----------
        path: str, optional
            If not None, save the plot as an png-image under this file name.
        """
        if self.history is None:
            print("ERROR: The training history is not available.")
            return

        if self.NO_PLOT: return

        n = np.arange(len(self.history["loss"]))

        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle("training loss and accuracy")
        ax1.plot(n, self.history["loss"], color="tab:blue", label="train_loss")
        ax1.plot(n, self.history["val_loss"], color="tab:orange", label="val_loss")
        ax1.set_ylabel("loss")
        ax1.set_yscale("log")
        ax1.legend(loc="upper right")
        ax2.plot(n, self.history["accuracy"], color="tab:green", label="train_acc")
        ax2.plot(n, self.history["val_accuracy"], color="tab:red", label="val_acc")
        ax2.set_ylim([-0.1, 1.1])
        ax2.set_ylabel("accuracy")
        ax2.legend(loc="lower left")
        plt.xlabel("epoch")
        if path != None:
            plt.savefig(path, format="png", dpi=300)
        plt.show()

    def get_optimizer(self, optimizer_name, learning_rate):

        """
        optimizer_name: string
            Name of the optimizer that is returned
        learning_rate: float
            Learning rate of the optimizer
        returns: optimizer
            The created Keras optimizer
        """
        if optimizer_name == "adam" and self.config.accum <= 1:
            return Adam(learning_rate=learning_rate)
        if optimizer_name == "sgd" and self.config.accum <= 1:
            return SGD(learning_rate=learning_rate)
        # Since we did not succeed in implementing a accumulationg Optimizer the following code is not longer active
        if optimizer_name == "adam" and self.config.accum > 1:
            print(
                f"Accumulating {self.config.accum} batches with AdamAccumulate with a resulting batch size of {self.config.batch_size * self.config.accum}")
            return opt.AdamAccumulate(steps=self.config.accum, lr=learning_rate)
        if optimizer_name == "sgd" and self.config.accum > 1 and False:
            return opt.AdamAccumulate(steps=self.config.accum, learning_rate=learning_rate)

    def get_pretrained(self):
        """
        returns: model
            returns pretrained Keras model
        """
        base_model = tf.keras.applications.EfficientNetB2(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=4,
            classifier_activation="softmax",
        )
        x = base_model.output

        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(4024, activation='relu')(x)
        x = Dropout(self.config.dropout)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(self.config.dropout)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(self.config.dropout)(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(self.config.output_nodes, activation='softmax')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        return model

    def save(self):
        """
        saves the current model under the name given in the model config
        """
        self.model.save(self.config.model_name)

    def load(self):
        """
        loads model when self.MODE == "continue" or "validate"
        returns: string
            returns self.Mode for access in main
        """
        if os.path.isdir(self.config.model_name) and self.MODE == "continue" or self.MODE == "validate":
            self.model = tf.keras.models.load_model(self.config.model_name)
            return self.MODE
        else:
            return self.MODE
