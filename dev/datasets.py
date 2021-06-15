import sys
import os
import time
import glob
from random import shuffle, randint

import mrcfile
import numpy as np
from PIL import Image
from skimage.exposure import equalize_adapthist as clahe
from skimage.transform import rescale
from torch.utils.data import Dataset, IterableDataset
import torch

class ImageData(Dataset):
    pass

"""
.mrc files tend to be unwieldy, so the following two classes provide four
methods for chunked loading via mrcfile's mmap (memory map, via numpy) feature.
"""

class MRCSampler:

    def __init__(self, path_to_mrc, tile_size, tile_stride,
            shuffled=False, verbose=False):
        self.path_to_mrc = path_to_mrc
        self.mmap = mrcfile.mmap(path_to_mrc)
        if verbose:
            print(path_to_mrc)
        self._depleted = False

        length, width, height = self.mmap.data.shape
        self.cols = (width // tile_stride)
        self.rows = (height // tile_stride)
        self.tile_size = tile_size
        self.tile_stride = tile_stride

        self._i = 0
        self.shuffled = shuffled
        if self.shuffled:
            self.I = list(range(len(self)))
            shuffle(self.I)

        self.verbose = verbose

    def __len__(self):
        return self.cols * self.rows

    # associates an index in range(len(self)) to a square tile in the .mrc
    def __getitem__(self, idx):
        col = idx // self.rows
        row = idx % self.rows
        l = col * self.tile_stride
        r = l + self.tile_size
        t = row * self.tile_stride
        b = t + self.tile_size
        if self.verbose:
            print(self._i,(l,r),(t,b))
        return self.mmap.data[:, l:r, t:b]

    def get_tile(self, k = 1):
        tile = []
        for _ in range(k):
            if self.shuffled:
                i = self.I[self._i]
            else:
                i = self._i
            tile.append(self.__getitem__(i))
            self._i += 1
            if self._i >= len(self):
                self._depleted = True
                if self.verbose:
                    print("i'm depleted :-(")
        if k == 1:
            return tile[0]
        else:
            return tile

    def is_depleted(self):
        return self._depleted

    def close(self):
        self.mmap.close()

    @classmethod
    def get_factory(cls, tile_size, tile_stride, shuffled, verbose):
        f = lambda path: cls(path, tile_size, tile_stride, shuffled, verbose)
        return f

class MRCData(IterableDataset):
    def __init__(self, source_path, sampler_factory,
            transform=None, K=1, shuffled=False, verbose=False, c=1):
        super(MRCData, self).__init__()
        self.transform = transform
        self.c = c
        self.K = K
        self.shuffled = shuffled
        self.verbose = verbose
        # method for generating MRCSampler objects
        self._sampler_factory = sampler_factory
        # glob all the mrc files in the source directory
        self.source_path = source_path
        if os.path.isdir(self.source_path):
            self.mrc_files = glob.glob(os.path.join(source_path, "*.mrc"))
        else:
            with open(self.source_path, 'r', newline='\n') as curated_paths:
                self.mrc_files = [x.replace('\n', '') for x in curated_paths.readlines()]
        self.nb_sources = len(self.mrc_files)
        if verbose:
            print("identified {} mrc files in {}".format(
                len(self.mrc_files),
                self.source_path
            ))
        self.mrc_samplers = []
        for path in self.mrc_files:
            self.mrc_samplers.append(self._sampler_factory(path))
            if verbose: print('loaded', path)

    def __len__(self):
        return sum([len(self.mrc_samplers[i]) for i in range(self.nb_sources)])

    def _refresh_sampler(self):
        if self.shuffled:
            self._sampler = self.mrc_samplers[randint(0,self.nb_sources-1)]
        else:
            self._sampler = self.mrc_samplers[self._loc]

        if self._sampler.is_depleted():
            if all([samp.is_depleted() for samp in self.mrc_samplers]):
                raise StopIteration
            self._loc = (self._loc + 1) % self.nb_sources
            self._refresh_sampler()

    def __next__(self):
        # go to the next sampler if this one is depleted
        # if shuffled, switch to a random (un-depleted)
        try:
            self._refresh_sampler()
        except StopIteration:
            for sampler in self.mrc_samplers:
                sampler.close()
            raise StopIteration
        # get a randomly-sampled tile from the image
        img = np.array(self._sampler.get_tile())
        img = np.sum(img, axis=0)
        if self.c > 1:
            rows,cols = img.shape
            img = img.reshape((rows,cols,1))
            img = np.repeat(img, self.c, axis=2)
        img = np.uint8(img)
        # conditionally generate K augmentations of the image
        # https://github.com/mpatacchiola/self-supervised-relational-reasoning/
        #pic = Image.fromarray(img)
        img_list = list()
        if self.transform is not None:
            for _ in range(self.K):
                img_transformed = self.transform(img.copy())
                img_list.append(img_transformed)
        else:
            img_list = img
        return img_list, 0

    def __iter__(self):
        self.depleted = False
        self._loc = 0 # file to sample from
        self._refresh_sampler()
        if torch.utils.data.get_worker_info() is not None:
            raise NotImplemented("per Multiprocessing Best Processes\nhttps://pytorch.org/docs/stable/notes/multiprocessing.html")
        return self

def main():
    iter_mode = int(sys.argv[4])
    if iter_mode == 0:
        sampler_shuffle = False
        dataset_shuffle = False
    elif iter_mode == 1:
        sampler_shuffle = True
        dataset_shuffle = False
    elif iter_mode == 2:
        sampler_shuffle = False
        dataset_shuffle = True
    elif iter_mode == 3:
        sampler_shuffle = True
        dataset_shuffle = True
    start = time.time()
    factory = MRCSampler.get_factory(tile_size = int(sys.argv[2]),
                                     tile_stride = int(sys.argv[3]),
                                     shuffled = sampler_shuffle,
                                     verbose = True)
    ds = MRCData(source_path = sys.argv[1],
                 sampler_factory = factory,
                 shuffled = dataset_shuffle,
                 verbose = False)
    for x in ds:
        z = x+1
    print(f"iterated {len(ds.mrc_files)} files in {time.time() - start} seconds.")

if __name__ == "__main__":
    main()
