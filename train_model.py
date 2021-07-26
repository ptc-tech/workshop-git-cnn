# -*- coding: utf-8 -*-
# @Author: Benjamin Cohen-Lhyver
# @Date:   2021-07-24 16:15:34
# @Last Modified by:   Benjamin Cohen-Lhyver
# @Last Modified time: 2021-07-25 19:15:42


##############################################################################
import argparse
import json
import CNNModel as cm
##############################################################################


##############################################################################
parser = argparse.ArgumentParser(description='Training of the model')

parser.add_argument('-e', '--epochs',
                    default=0,
                    type=int,
                    help='number of training epochs')


args = parser.parse_args()
##############################################################################


##############################################################################
if args.epochs != 0:

    with open("parameters.json") as params_file:
        parameters = json.load(params_file)

    parameters["nb_epochs"] = args.epochs

    with open("parameters.json") as params_file:
        json.dump(parameters, params_file)
##############################################################################


##############################################################################
model = cm.CNNModel()

model.import_data()

model.create_model()

model.train_model()

model.save_history()
##############################################################################



#######################################################################################################################

                                         # === END OF FILE === #

#######################################################################################################################