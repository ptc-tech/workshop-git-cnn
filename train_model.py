# -*- coding: utf-8 -*-
# @Author: Benjamin Cohen-Lhyver
# @Date:   2021-07-24 16:15:34
# @Last Modified by:   Benjamin Cohen-Lhyver
# @Last Modified time: 2021-07-25 08:29:08

import CNNModel as cm


model = cm.CNNModel()

model.import_data()

model.create_model()

model.train_model()

model.save_history()
