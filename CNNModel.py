# -*- coding: utf-8 -*-
# @Author: Benjamin Cohen-Lhyver
# @Date:   2021-07-21 10:45:04
# @Last Modified by:   Benjamin Cohen-Lhyver
# @Last Modified time: 2021-07-26 06:26:14


##############################################################################
import json
import pickle

# --- Manipulation de données
import numpy as np
import pandas as pd

# --- Display
# from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sn

# --- Machine Learning
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- App
# from flask import Flask
# from flask import jsonify, make_response, request
##############################################################################


##############################################################################

class CNNModel():
    """..."""

    def __init__(self):
        """..."""

        # --- Import des paramètres
        with open("./parameters.json") as f:
            self.parameters = json.load(f)

        self.model_trained = False


    def import_data(self):
        # --- Import des données
        (self.images_train, self.labels_train), (self.images_test, self.labels_test) = keras.datasets.mnist.load_data()

        self.images_train = self.images_train.astype("float32") / 255
        self.images_train = np.expand_dims(self.images_train, -1)

        self.images_test = self.images_test.astype("float32") / 255
        self.images_test = np.expand_dims(self.images_test, -1)

        self.nb_images = len(self.images_train)
        self.input_shape = (28, 28, 1)

        num_classes = len(np.unique(self.labels_train))

        self.labels_train = keras.utils.to_categorical(self.labels_train, num_classes)
        self.labels_test = keras.utils.to_categorical(self.labels_test, num_classes)

    # -------
    # def test_parameters_format(self):
    #     """..."""

    #     for iParam, iValue in self.parameters.items():

    #         if iParam not in ["activation_function"]:

    #             if type(iValue) not in [int, float]:
    #                 print(f"Error, parameter {iParam} with value {iValue} is not in the right format")
    #                 return 0

    #         else:

    #             if type(iValue) is not str:
    #                 print(f"Error, parameter {iParam} with value {iValue} is not in the right format")
    #                 return 0

    #     return 1


    def test_training_performance(self, thr=0.9):
        """..."""
        if self.model_trained:
            if self.model.history.history[-1] > thr:
                return 1

        return 0


    # ====================================== #

    #      2. VISUALISATION DES INPUTS

    def plot_random_inputs(self):
        """..."""
        random_indices = np.random.randint(0, self.nb_images, 25)

        plt.figure(figsize=(10,10))

        for iCpt, iIdx in enumerate(random_indices):
            plt.subplot(5, 5, iCpt)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.images_train[iIdx][:, :, 0], cmap=plt.cm.binary)
            plt.xlabel(self.labels_train[iIdx]);

        plt.show()

    # ====================================== #

    # ====================================== #

    #      1. CREATION DU MODELE CNN

    def create_model(self):
        """..."""
        self.model = keras.Sequential(
            [
             layers.Input(shape=self.input_shape),
             layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
             layers.MaxPooling2D(pool_size=(2, 2)),
             layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
             layers.MaxPooling2D(pool_size=(2, 2)),
             layers.Flatten(),
             layers.Dropout(self.parameters["last_dropout"]),
             layers.Dense(10, activation='softmax')
            ]
        )

        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    # ====================================== #

    # ====================================== #

    #      4. TRAINING

    def train_model(self):
        """..."""
        batch_size = self.parameters["batch_size"]
        nb_epochs = self.parameters["nb_epochs"]
        validation_split = self.parameters["validation_split"]

        self.history = self.model.fit(
            self.images_train,
            self.labels_train,
            batch_size=batch_size,
            epochs=nb_epochs,
            validation_split=validation_split)


    def save_history(self):
        """..."""

        with open("training_history.pkl", "wb") as outfile:
            pickle.dump(self.model.history.history, outfile)

    # ====================================== #

    # ====================================== #

    #      2. VISUALISATION DES RESULTATS

    def plot_results(self):
        """..."""
        pd.DataFrame(self.history.history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
        plt.show()

    # ====================================== #

    # ====================================== #

    #      5. EVALUATION & PRÉDICTION



    # ====================================== #

    # ====================================== #

    # #      6. API

    # app = Flask(__name__)


    # if __name__ == "__main__":
    #     app.run()
    # ====================================== #



# #######################################################################################################################

#                                          # === END OF FILE === #

# #######################################################################################################################
