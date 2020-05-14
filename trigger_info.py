import csv
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import cv2
import os

files = [f for f in os.listdir('vicon') if f.endswith('-Trig.csv')]

for fidx in range(len(files)):
    file = files[fidx]
    trig_list = []
    trig = False
    with open('vicon/{0}'.format(file)) as f:
        r = csv.reader(f)
        [next(r) for i in range(5)]
        for row in r:
            if row:
                if not trig and float(row[2]) < -0.2:
                    trig_list.append(int(row[0]))
                    trig = True
                elif trig and float(row[2]) > -0.2:
                    trig = False

    trig_list = np.array(trig_list)
    start = trig_list[:-1]
    end = trig_list[1:]
    delta = end - start
    name = file.split('-')[0]
    meanval = np.mean(delta)
    stdval = np.std(delta)
    maxval = np.max(delta)
    plt.subplot(3,5,fidx+1),plt.plot(delta),plt.title('{0} {1} - {2:.2f},{3:.2f},{4:.2f}'.format(name,trig_list.shape[0],meanval,stdval,maxval))
    print(file,trig_list.shape[0],delta[0],delta[-1],start[0],end[0],start[-1],end[-1],sep='\t')
plt.show()
"""
for file in os.listdir('vicon_proc'):
    frame_data = np.load('vicon_proc/{0}'.format(file))
    print(file,'  \t',frame_data.shape[0])"""
