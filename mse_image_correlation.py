import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

images = 255*np.load('vera_proc/BallGrab-200205-01-images.npy')
angles = np.load('vicon_proc/BallGrab-200205-01-angles.npy')
#lines = np.expand_dims(lines, axis=3)

frames = [Image.fromarray(frame) for frame in images]

frames[0].save('test.gif', save_all=True, append_images=frames[1:], duration=1/25, loop=0)

mse = [np.mean(np.power(frame - images[0], 2)) for frame in images]
plt.subplot(1,2,1),plt.plot(mse),plt.plot(30+20*angles[:,0])
#plt.axvline(x=155, color='r')
plt.show()
