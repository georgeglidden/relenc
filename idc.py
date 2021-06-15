"""
image data curator: a minimal command line tool for fast manual iteration of
image-like datasets via a sampling rule and a presentation function.
idc supports standard image formats (.jpeg, .png, .tif, etc) as well
as the .mrc format.
"""

import os
import sys
import glob

import matplotlib.pyplot as plt
from skimage.transform import rescale
from skimage.exposure import equalize_adapthist
from numpy import mean,sum

from dev import MRCSampler

def norm_minmax(x):
    x -= x.min()
    return x / x.max()

def mrc_to_img(imlike, scale=0.1):
    x = sum(imlike, axis=0)
    x = norm_minmax(x)
    x = equalize_adapthist(x)
    return rescale(x, scale)

def present(imlike, l):
    im = l(imlike)
    plt.imshow(im)
    plt.show(block=False)

def main():
    pattern = sys.argv[1]
    size = int(sys.argv[2])
    stride = int(sys.argv[3])
    k = int(sys.argv[4])
    matches = glob.glob(pattern)
    selected = []
    print(f"identified {len(matches)} files")
    input()
    for path in matches:
        dot_loc = path.rfind('.')
        ext = path[dot_loc:]
        if ext == '.mrc':
            sampler = MRCSampler(path, size, stride)
            samples = sampler.get_tile(k)
            print(len(sampler), sampler.rows, sampler.cols, end='\n>')
            for x in samples:
                present(x, l=lambda imlike: mrc_to_img(imlike))
                if (input()+'_')[0].lower() == 'y':
                    selected.append(path)
                    break
        else:
            im = imread(path)
            present(im, l=lambda x: x)
            plt.show(block=False)
    dir, _ = os.path.split(pattern)
    record_path = os.path.join(dir, "selected_files.txt")
    with open(record_path, 'w+') as record_file:
        record_file.write('\n'.join(selected))
    print(record_path)
    exit()

if __name__ == '__main__':
    main()
    plt.show()
