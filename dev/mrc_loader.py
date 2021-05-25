import sys
import os
import time
import glob
from random import shuffle, randint

import mrcfile
from numpy import sum,mean
from PIL import Image
from skimage.exposure import equalize_adapthist as clahe
from skimage.transform import rescale
from torch.utils.data import IterableDataset
import torch

class MRCSampler:

    def __init__(self, path_to_mrc, tile_size, tile_stride,
            shuffled=False, verbose=False):
        self.path_to_mrc = path_to_mrc
        self.mmap = mrcfile.mmap(path_to_mrc)
        self._depleted = False

        length, width, height = self.mmap.data.shape
        self.cols = (width // tile_stride)
        self.rows = (height // tile_stride)
        self.tile_size = tile_size
        self.tile_stride = tile_stride

        self._i = 0
        self.I = list(range(len(self)))
        if shuffled: shuffle(self.I)
        self.shuffled = shuffled

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

    def get_tile(self):
        tile = self.__getitem__(self.I[self._i])
        self._i += 1
        if self._i >= len(self):
            self._depleted = True
            if self.verbose:
                print("i'm depleted :-(")
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
    def __init__(self, source_dir, sampler_factory,
            transform=None, K=1, shuffled=False, verbose=False):
        super(MRCData, self).__init__()
        self.transform = transform
        self.K = K
        self.shuffled = shuffled
        self.verbose = verbose
        # method for generating MRCSampler objects
        self._sampler_factory = sampler_factory
        # glob all the mrc files in the source directory
        self.source_dir = source_dir
        self.mrc_files = glob.glob(os.path.join(source_dir, "*.mrc"))
        self.nb_sources = len(self.mrc_files)
        if verbose:
            print("identified {} mrc files in {}".format(
                len(self.mrc_files),
                self.source_dir
            ))
        self.mrc_samplers = [self._sampler_factory(path) for path in self.mrc_files]

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
            for sampler in mrc_samplers:
                sampler.close()
            raise StopIteration
        # get a randomly-sampled tile from the image
        img = self._sampler.get_tile()

        return img
        # conditionally generate K augmentations of the image
        # see https://github.com/mpatacchiola/self-supervised-relational-reasoning/
        # pic = Image.fromarray(img)
        # img_list = list()
        # if self.transform is not None:
        #     for _ in range(self.K):
        #         img_transformed = self.transform(pic.copy())
        #         img_list.append(img_transformed)
        # else:
        #     img_list = img
        # print(self._idx, self._sampler.source_file)
        # return img_list

    def __iter__(self):
        self.depleted = False
        self._loc = 0 # file to sample from
        self._refresh_sampler()
        if torch.utils.data.get_worker_info() is not None:
            raise NotImplemented("TODO - distributed MMMs (memory mapped .mrc files) among workers")
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
    ds = MRCData(source_dir = sys.argv[1],
                 sampler_factory = factory,
                 shuffled = dataset_shuffle,
                 verbose = False)
    for x in ds:
        z = x+1
    print(f"iterated {len(ds.mrc_files)} files in {time.time() - start} seconds.")

if __name__ == "__main__":
    main()
