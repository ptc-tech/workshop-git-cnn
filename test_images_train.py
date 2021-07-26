# -*- coding: utf-8 -*-
# @Author: Benjamin Cohen-Lhyver
# @Date:   2021-07-24 10:15:59
# @Last Modified by:   Benjamin Cohen-Lhyver
# @Last Modified time: 2021-07-25 19:16:26

import pickle

# if __name__ == "__main__":

with open("training_history.csv", "rb") as infile:
    history = pickle.load(infile)

assert history["accuracy"][-1] < 0.95
assert history["loss"] < 0.1
