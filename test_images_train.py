# -*- coding: utf-8 -*-
# @Author: Benjamin Cohen-Lhyver
# @Date:   2021-07-24 10:15:59
# @Last Modified by:   Benjamin Cohen-Lhyver
# @Last Modified time: 2021-07-26 06:23:55

import pickle

# if __name__ == "__main__":

with open("training_history.pkl", "rb") as history_file:
    history = pickle.load(history_file)

assert history["accuracy"][-1] > 0.95
assert history["loss"][-1] < 0.1
