# -*- coding: utf-8 -*-
# @Author: Benjamin Cohen-Lhyver
# @Date:   2021-07-24 10:15:59
# @Last Modified by:   Benjamin Cohen-Lhyver
# @Last Modified time: 2021-07-25 08:27:24

import pickle

with open("training_history.csv", "rb") as infile:
    history = pickle.load(infile)


if history.accuracy[-1] < 0.95 and history.loss > 0.1:
    return 0

return 1