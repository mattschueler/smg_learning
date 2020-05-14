import csv
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import cv2
import os

with open("filtered_mocap.csv", "r") as f:
    r = csv.reader(f)
    i=0
    for row in r:
        for j in range(5):
            for k in range(4):
                try:
                    start = 2+3*k+12*j
                    finger_data[i,j,k] = row[start:start+3]
                except:
                    print(i,j,k)
        i += 1

finger_data = finger_data.astype(float)
print(finger_data.shape)

angles = np.zeros((1366,5))
for j in range(5):
    vectorA = np.subtract(finger_data[:,j,1,:],finger_data[:,j,0,:])
    vectorB = np.subtract(finger_data[:,j,3,:],finger_data[:,j,2,:])
    dotProd = np.einsum('ij, ij->i', vectorA, vectorB)
    vectorAnorm = LA.norm(vectorA, 2, axis=1)
    vectorBnorm = LA.norm(vectorB, 2, axis=1)
    magProduct = np.multiply(vectorAnorm, vectorBnorm)
    cosVals = np.divide(dotProd, magProduct)
    angles[:,j] = np.arccos(cosVals)

np.save('ang_train.npy', angles)

us_data = np.zeros((1366,162816))
with open("filtered_us.csv", "r") as f:
    r = csv.reader(f)
    i=0
    for row in r:
        us_data[i,:] = row

us_data.reshape((1366,636,256), order='F')
us_data = cv2.normalize(us_data,None,0,1,cv2.NORM_MINMAX)
np.save('us_train.npy', us_data)
