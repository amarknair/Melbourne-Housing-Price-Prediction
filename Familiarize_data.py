#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: amar.kelunair
"""
import os
dirname = os.path.dirname(__file__)

import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

melb_data = pd.read_csv(dirname+'/input/melbourne-housing-snapshot/panda_test.csv')
print(melb_data.describe())
print(melb_data.columns)