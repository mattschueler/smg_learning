import os
import numpy as np
import csv
import cv2

trials = [f for f in os.listdir('vera_csv') if f.endswith('.csv')]

for trial in trials:
    print(trial)
    us_data = np.zeros((1399,162816))
    print('opening')
    with open('vera_csv/{0}'.format(trial)) as f:
        r = csv.reader(f)
        i=0
        for row in r:
            us_data[i,:] = row
            i += 1
    print('reshaping')
    us_data = us_data.reshape((1399,636,256), order='F')
    print('normalizing')
    us_data = cv2.normalize(us_data,None,0,1,cv2.NORM_MINMAX)
    data_pow = np.power(us_data,1/5)
    data_norm = (data_pow-np.mean(data_pow))/np.std(data_pow)
    print('saving')
    np.save('vera_proc/{0}-images.npy'.format(trial.split('.')[0]), data_norm)
