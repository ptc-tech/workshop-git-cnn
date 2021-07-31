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

# initlize variables
model = None
history = None
# ====================================== #

#      2. VISUALISATION DES INPUTS



# ====================================== #

# ====================================== #

#      1. CREATION DU MODELE CNN
def create_model():
    model = keras.Sequential(
        [
         layers.Input(shape=(32, 32, 3)),
         layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
         layers.MaxPooling2D(pool_size=(2, 2)),
         layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
         layers.MaxPooling2D(pool_size=(2, 2)),
         layers.Flatten(),
         layers.Dropout(0.5),
         layers.Dense(10, activation='softmax')
        ]
    )

    model.compile(optimizer="Adam", loss="mse", metrics=["acc"])

    return "<p>model built!</p>"

# ====================================== #

# ====================================== #

#      4. TRAINING
def train_model():
    if model is None:
        return "<p>you should create the model first using the rout /create_model</p>"
    else:
        batch_size = parameters["batch_size"]
        nb_epochs = parameters["epochs"]
        validation_split = 0.2


        history = model.fit(images_train, labels_train, batch_size=batch_size, epochs=nb_epochs, validation_split= validation_split)


# ====================================== #

# ====================================== #

#      2. VISUALISATION DES RESULTATS
def visualize_result():
    return "<p></p>"


# ====================================== #

# ====================================== #

#      5. EVALUATION & PRÉDICTION

def predict(image):
    if history is None:
        return "<p>you should train the model first using the rout /create_model</p>"
    else:
        return ""
# part of the prediction

# ====================================== #
test_loss, test_acc = model.evaluate(images_test, labels_test, verbose=2)
predictions = model.predict(images_test)
# ====================================== #

#      6. API

app = Flask(__name__)

@app.route("/create_model")
def call_create_model():
    return create_model()

@app.route("/train_model")
def call_train_model():
    return train_model()

@app.route("/visualize_result")
def call_visualize_result():
    return visualize_result()

@app.route("/predict")
def call_predict(image):
    return predict(image)

@app.route('/')
def index():
    return 'index'

if __name__ == "__main__":
    app.run()
# ====================================== #



# #######################################################################################################################

#                                          # === END OF FILE === #

# #######################################################################################################################