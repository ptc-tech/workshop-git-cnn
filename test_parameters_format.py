# -*- coding: utf-8 -*-
# @Author: Benjamin Cohen-Lhyver
# @Date:   2021-07-25 10:27:33
# @Last Modified by:   Benjamin Cohen-Lhyver
# @Last Modified time: 2021-07-26 06:25:57


import json


def test_format():
    """..."""

    with open("parameters.json") as inflile:
        parameters = json.load(infile)

    for iParam, iValue in parameters.items():

        if iParam in ["activation_function"]:
            assert type(iValue) is not str

        else:
            assert type(iValue) not in [int, float]