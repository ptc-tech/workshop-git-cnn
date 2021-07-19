
##############################################################################
# --- Manipulation de données
import numpy as np
import pandas as pd

# --- Display
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sn

# --- Machine Learning
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- App
from flask import Flask
from flask import jsonify, make_response, request
##############################################################################


##############################################################################
(images_train, labels_train), (images_test, labels_test) = keras.datasets.mnist.load_data()


test = 3
# ====================================== #

#      2. VISUALISATION DES INPUTS



# ====================================== #

# ====================================== #

#      1. CREATION DU MODELE CNN



# ====================================== #

# ====================================== #

#      4. TRAINING

batch_size = parameters["batch_size"]
nb_epochs = parameters["epochs"]
validation_split = 0.2

history = model.fit(images_train, labels_train, batch_size=batch_size, epochs=nb_epochs, validation_split= validation_split)


# ====================================== #

# ====================================== #

#      2. VISUALISATION DES RESULTATS



# ====================================== #

# ====================================== #

#      5. EVALUATION & PRÉDICTION



# ====================================== #

# ====================================== #

#      6. API

app = Flask(__name__)


if __name__ == "__main__":
    app.run()
# ====================================== #



# #######################################################################################################################

#                                          # === END OF FILE === #

# #######################################################################################################################
