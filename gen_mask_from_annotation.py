import numpy as np
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import os, sys, shutil, argparse
import openslide
import json
from scipy import ndimage as ndi

parser = argparse.ArgumentParser(description='Get tissue mask of WSI and save it in npy format')
parser.add_argument('--wsi_path', default='./data/tif_wsi/test_001.tif', metavar='TIF_PATH', type=str, help='Path to the original tif file')
parser.add_argument('--json_path', default='./jsons/test_001.json', metavar='JSON_PATH', type=str, help='Path to annotation json file (created from xml)')
parser.add_argument('--level', default=6, type=int, help='at which WSI level to obtain the mask, default 6')
parser.add_argument('--mask_path', default='./data/png_mask_files/test_001.png', metavar='MASK_PNG_PATH', type=str, help='Path to mask png file')

args = parser.parse_args()

path =args.json_path
file = open(path, 'r')
jdict = json.load(file)

positives =  jdict['positive']
negatives =  jdict['negative']
# print(len(positives))
# print(len(negatives))

def run(args):
    scale = 2**(args.level)
    slide = openslide.OpenSlide(args.wsi_path)
    # note the shape of img_RGB is the transpose of slide.level_dimensions
    img_RGB = np.transpose(
        np.array(slide.read_region((0, 0),args.level, slide.level_dimensions[args.level]).convert('RGB')),
        axes=[1, 0, 2]
    )
    #level 6 mask shape
    mh, mw = img_RGB.shape[0], img_RGB.shape[1]
    mask_arr = np.zeros((mh, mw))
    # print('mask_arr of shape', mask_arr.shape)
    # print('img_RGB is of shape', mask_arr.shape)
    for i in range(len(positives)):
        outline = np.asarray(positives[i]['vertices']) // scale
        # print(outline)
        x = outline[:,0]
        y = outline[:,1]
        mask_arr[x,y] = 1.0
    #not sure if I should do this, but ignore the negetives
    t1 = ndi.binary_dilation(mask_arr)
    t2 = ndi.binary_fill_holes(t1)
    plt.figure(figsize=(10,10)) 
    plt.imshow(t2, cmap='gray')
    plt.savefig(args.mask_path)
    plt.close()

def main():
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()