import csv
import os
import numpy as np
from numpy import linalg as LA

finger_data = np.zeros((1366, 5, 4, 3))

trials = ['-'.join(f.split('-')[:-1]) for f in os.listdir('vicon') if f.endswith('-Trig.csv')]
for trial in trials:
    print(trial)
    print('reading triggers')
    with open('vicon/{0}-Trig.csv'.format(trial)) as f:
        r = csv.reader(f)
        [next(r) for i in range(5)]
        trig = False
        trig_list = []
        for row in r:
            if row:
                if not trig and float(row[2]) < -0.2:
                    trig_list.append(row[0])
                    trig = True
                elif trig and float(row[2]) > -0.2:
                    trig = False

    print('reading frames')
    with open('vicon/{0}-Mocap.csv'.format(trial)) as f:
        r = csv.reader(f)
        [next(r) for i in range(5)]
        frame_data = []
        for row in r:
            if row and row[0] in trig_list:
                frame_data.append(row[2:])

    print('reformatting')
    finger_data = np.zeros((len(frame_data),5,4,3))
    for i in range(len(frame_data)):
        for j in range(5):
            for k in range(4):
                for m in range(3):
                    finger_data[i,j,k,m] = frame_data[i][12*j+3*k+m]

    print('calculating angles')
    vectorA = np.subtract(finger_data[:,:,1,:],finger_data[:,:,0,:])
    vectorB = np.subtract(finger_data[:,:,3,:],finger_data[:,:,2,:])
    dotProd = np.einsum('ijk, ijk->ij', vectorA, vectorB)
    vectorAnorm = LA.norm(vectorA, 2, axis=2)
    vectorBnorm = LA.norm(vectorB, 2, axis=2)
    magProduct = np.multiply(vectorAnorm, vectorBnorm)
    cosVals = np.divide(dotProd, magProduct)
    angles = np.arccos(cosVals)

    angles = np.zeros((len(frame_data),5))
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

    print('saving data')
    np.save('vicon_proc/{0}-angles.npy'.format(trial), angles)
