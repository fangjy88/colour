# -*- coding: utf-8 -*-
"""
Created on Tue Mar 08 10:20:14 2016

@author: Fang

data measured by Fang
algorithm referenced by <Computaional Colour Science Using MATLAB>


"""

import pandas as pd
import numpy as np

filedir = '.\dataset\DC_D3x_data.csv'
csvdf = pd.read_csv(filedir)

rgb = csvdf.loc[:,['R','G','B']].as_matrix()
xyz = csvdf.loc[:,['X','Y','Z']].as_matrix()
wxyz = csvdf.loc[0,['WX','WY','WZ']].as_matrix()

rgb_train = rgb[::2,:]
xyz_train = xyz[::2,:]

rgb_test = rgb[1::2,:]
xyz_test = xyz[1::2,:]

# find the linear transform between rgb_train and xyz_train

M = np.dot(xyz_train.transpose(),np.linalg.pinv(rgb_train.transpose()))
xyz_test_pred = np.dot(M,rgb_test.transpose()).transpose()
# calculate color difference

# now perform a second-order transform

#now perform a third-order transform