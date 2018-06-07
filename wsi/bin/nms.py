import sys
import os
import argparse

import numpy as np
from skimage import filters

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')


parser = argparse.ArgumentParser(description='Generate predicted coordinates'
                                 ' from probability map of tumor patch'
                                 ' predictions, using non-maximal suppression')
parser.add_argument('probs_map_path', default=None, metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the input probs_map numpy file')
parser.add_argument('coord_path', default=None, metavar='COORD_PATH',
                    type=str, help='Path to the output coordinates csv file')
parser.add_argument('--level', default=6, type=int, help='at which WSI level'
                    ' the probability map was generated, default 6,'
                    ' i.e. inference stride = 64')
parser.add_argument('--radius', default=12, type=int, help='radius for nms,'
                    ' default 12 (6 used in Google paper at level 7,'
                    ' i.e. inference stride = 128)')
parser.add_argument('--prob_thred', default=0.5, type=float,
                    help='probability threshold for stopping, default 0.5')
parser.add_argument('--sigma', default=0.0, type=float,
                    help='sigma for Gaussian filter smoothing, default 0.0,'
                    ' which means disabled')


def run(args):
    probs_map = np.load(args.probs_map_path)
    X, Y = probs_map.shape
    resolution = pow(2, args.level)

    if args.sigma > 0:
        probs_map = filters.gaussian(probs_map, sigma=args.sigma)

    outfile = open(args.coord_path, 'w')
    while np.max(probs_map) > args.prob_thred:
        prob_max = probs_map.max()
        max_idx = np.where(probs_map == prob_max)
        x_mask, y_mask = max_idx[0][0], max_idx[1][0]
        x_wsi = int((x_mask + 0.5) * resolution)
        y_wsi = int((y_mask + 0.5) * resolution)
        outfile.write('{:0.5f},{},{}'.format(prob_max, x_wsi, y_wsi) + '\n')

        x_min = x_mask - args.radius if x_mask - args.radius > 0 else 0
        x_max = x_mask + args.radius if x_mask + args.radius <= X else X
        y_min = y_mask - args.radius if y_mask - args.radius > 0 else 0
        y_max = y_mask + args.radius if y_mask + args.radius <= Y else Y

        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                probs_map[x, y] = 0

    outfile.close()


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
