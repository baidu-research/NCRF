import numpy as np

probmaps_dir = './sample_data/probmaps/'
a = np.load(probmaps_dir+'test_012.npy')
print(a.shape)
print(np.max(a), np.min(a))

