# -*- coding: utf-8 -*-
# @Author: Benjamin Cohen-Lhyver
# @Date:   2021-07-24 10:15:59
# @Last Modified by:   Benjamin Cohen-Lhyver
# @Last Modified time: 2021-07-25 19:55:39

import pickle

# if __name__ == "__main__":

with open("training_history.pkl", "rb") as history_file:
    history = pickle.load(history_file)

assert history["accuracy"][-1] > 0.95
assert history["loss"] < 0.1
