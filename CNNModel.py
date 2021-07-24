# -*- coding: utf-8 -*-
# @Author: Benjamin Cohen-Lhyver
# @Date:   2021-07-21 10:45:04
# @Last Modified by:   Benjamin Cohen-Lhyver
# @Last Modified time: 2021-07-24 08:44:36


##############################################################################
import json

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


    def import_data(self):
        # --- Import des données
        (self.images_train, self.labels_train), (self.images_test, self.labels_test) = keras.datasets.mnist.load_data()
        self.nb_images = len(images_train)

    # -------
    def test_parameters_format(self):
        """..."""

        for iParam, iValue in self.parameters.items():

            if iParam not in ["activation_function"]:

                if type(iValue) not in [int, float]:
                    print(f"Error, parameter {iParam} with value {iValue} is not in the right format")
                    return 0

            else:

                if type(iValue) is not str:
                    print(f"Error, parameter {iParam} with value {iValue} is not in the right format")
                    return 0

        return 1

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
        self.model = keras.Sequential(
            [
             keras.Input(shape=(32, 32, 3)),
             layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
             layers.MaxPooling2D(pool_size=(2, 2)),
             layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
             layers.MaxPooling2D(pool_size=(2, 2)),
             layers.Flatten(),
             layers.Dropout(0.5),
             layers.Dense(10, activation='softmax')
            ]
        )

        self.model.compile(optimizer="Adam", loss="mse", metrics=["acc"])

    # ====================================== #

    # ====================================== #

    #      4. TRAINING

    def train_model(self):
        """..."""
        batch_size = self.parameters["batch_size"]
        nb_epochs = self.parameters["epochs"]
        validation_split = self.parameters["validation_split"]

        self.history = self.model.fit(
            self.images_train,
            self.labels_train,
            batch_size=batch_size,
            epochs=nb_epochs,
            validation_split=validation_split)


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
