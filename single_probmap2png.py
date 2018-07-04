import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os, sys, shutil

probmaps_dir = './sample_data/'
png_dir = './sample_data/'
#try:
#   os.makedirs(png_dir)
#except:
#   pass

#l_probmaps = os.listdir(probmaps_dir)
#print(l_probmaps)
#for pm in l_probmaps:
pm = 'tumor_001_probmap_aaf.npy'
fname = probmaps_dir+pm
t = np.load(fname)
m1, m2 = np.min(t), np.max(t)
plt.figure()
plt.imshow(t, cmap='gray', vmin=m1, vmax=m2)
plt.colorbar()
plt.savefig(png_dir+pm.replace('npy','png'))
plt.close()
