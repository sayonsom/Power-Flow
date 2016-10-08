# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:43:37 2016

@author: sayonsomchanda
"""

import pandas as pd
import numpy as np

raw = []
with open('ieee14cdf.txt','r') as f:
    for line in f:
        raw.append(line.split())
data = pd.DataFrame(raw,columns = ['row','column','value'])
data_ind = data.set_index(['row','column']).unstack('column')
np.array(data_ind.values,dtype=float)