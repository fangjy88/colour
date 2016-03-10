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

# notice that the xyz is not nomalized
# calculate color difference
lab_test_pred = colour.models.cie_lab.XYZ_to_Lab(xyz_test_pred)
lab_test = colour.models.cie_lab.XYZ_to_Lab(xyz_test)
error = colour.difference.delta_e.delta_E_CIE1976(lab_test,lab_test_pred)
error_av = np.average(error)

print('mean colour differences for linear transform')
print(error_av)


# now perform a second-order transform
# [r, g, b, r^2, g^2, b^2, r^g, r^b, g^b, ones]
rgb_train2 = np.column_stack((rgb_train, rgb_train[:,0]**2, rgb_train[:,1]**2,\
                       rgb_train[:,2]**2, rgb_train[:,0]*rgb_train[:,1],\
                       rgb_train[:,0]*rgb_train[:,2], rgb_train[:,1]*rgb_train[:,2],\
                       np.ones(rgb_train.shape[0])))
rgb_test2= np.column_stack((rgb_test, rgb_test[:,0]**2, rgb_test[:,1]**2,\
                       rgb_test[:,2]**2, rgb_test[:,0]*rgb_test[:,1],\
                       rgb_test[:,0]*rgb_test[:,2], rgb_test[:,1]*rgb_test[:,2],\
                       np.ones(rgb_test.shape[0])))
M2 = np.dot(xyz_train.transpose(),np.linalg.pinv(rgb_train2.transpose()))
xyz_test_pred2 = np.dot(M2, rgb_test2.transpose()).transpose()
# notice that the xyz is not nomalized
# calculate color difference
lab_test_pred2 = colour.models.cie_lab.XYZ_to_Lab(xyz_test_pred2)
lab_test = colour.models.cie_lab.XYZ_to_Lab(xyz_test)
error2 = colour.difference.delta_e.delta_E_CIE1976(lab_test, lab_test_pred2)
error_av2 = np.average(error2)

print('mean colour differences for 2nd order transform')
print(error_av2)


#now perform a third-order transform
# [r, g, b, r^2, g^2, b^2, r^g, r^b, g^b, 
# r^3, g^3, b^3, r^2g, r^2b, g^2r, g^2b, b^2r, b^2g, ones]
rgb_train3 = np.column_stack((rgb_train, rgb_train[:,0]**2, rgb_train[:,1]**2,\
                       rgb_train[:,2]**2, rgb_train[:,0]*rgb_train[:,1],\
                       rgb_train[:,0]*rgb_train[:,2], rgb_train[:,1]*rgb_train[:,2],\
                       rgb_train[:,0]**3, rgb_train[:,1]**3, rgb_train[:,2]**3,\
                       rgb_train[:,0]**2*rgb_train[:,1], rgb_train[:,0]**2*rgb_train[:,2],\
                       rgb_train[:,1]**2*rgb_train[:,0], rgb_train[:,1]**2*rgb_train[:,2],\
                       rgb_train[:,2]**2*rgb_train[:,0], rgb_train[:,2]**2*rgb_train[:,1],\
                       np.ones(rgb_train.shape[0])))
rgb_test3 = np.column_stack((rgb_test, rgb_test[:,0]**2, rgb_test[:,1]**2,\
                       rgb_test[:,2]**2, rgb_test[:,0]*rgb_test[:,1],\
                       rgb_test[:,0]*rgb_test[:,2], rgb_test[:,1]*rgb_test[:,2],\
                       rgb_test[:,0]**3, rgb_test[:,1]**3, rgb_test[:,2]**3,\
                       rgb_test[:,0]**2*rgb_test[:,1], rgb_test[:,0]**2*rgb_test[:,2],\
                       rgb_test[:,1]**2*rgb_test[:,0], rgb_test[:,1]**2*rgb_test[:,2],\
                       rgb_test[:,2]**2*rgb_test[:,0], rgb_test[:,2]**2*rgb_test[:,1],\
                       np.ones(rgb_test.shape[0])))
M3 = np.dot(xyz_train.transpose(),np.linalg.pinv(rgb_train3.transpose()))
xyz_test_pred3 = np.dot(M3, rgb_test3.transpose()).transpose()
# notice that the xyz is not nomalized
# calculate color difference
lab_test_pred3 = colour.models.cie_lab.XYZ_to_Lab(xyz_test_pred3)
lab_test = colour.models.cie_lab.XYZ_to_Lab(xyz_test)
error3 = colour.difference.delta_e.delta_E_CIE1976(lab_test, lab_test_pred3)
error_av3 = np.average(error3)

print('mean colour differences for 3rd order transform')
print(error_av3)

