import json
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Get tissue mask of WSI and save it in npy format')
parser.add_argument('json_path', default=None, metavar='NPY_PATH', type=str, help='Path to the json file')
args = parser.parse_args()

path =args.json_path

file = open(path, 'r')
jdict = json.load(file)

plt.figure(figsize=(10,10))
scale = 32

positives =  jdict['positive']
negatives =  jdict['negative']
for i in range(len(positives)):
	outline = np.asarray(positives[i]['vertices']) // scale

	x = outline[:,0]
	y = outline[:,1]
	_ = plt.plot(x, y,'o', color='#f16824')

for i in range(len(negatives)):
	outline = np.asarray(negatives[i]['vertices']) // scale

	x = outline[:,0]
	y = outline[:,1]
	_ = plt.plot(x,y,'o', color='#000000')

# plt.xlim([0, 97792 // scale])
# plt.ylim([0, 219648 // scale])
plt.show()
