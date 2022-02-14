from pathlib import Path

import cupy as cp
import numpy as np

from pykilosort.utils import RawDataLoader, Bunch
from pykilosort.preprocess import gpufilter
from ibllib.dsp.voltage import destripe

BATCH_LENGTH = 65600

class CustomDataLoader(RawDataLoader):

    def __init__(self, *args, **kwargs):
        if 'whitening_matrix' in kwargs.keys():
            self.whitening_matrix = kwargs.pop('whitening_matrix')
        else:
            self.whitening_matrix = None
        super(CustomDataLoader, self).__init__(*args, **kwargs)


    def load_batch(self, batch_number):
        """
        Loads a batch of data to the GPU
        :param batch_number: Specifies which batch to load
        :return: Loaded batch
        """
        batch_data = self.load(batch_number * BATCH_LENGTH, (batch_number+1) * BATCH_LENGTH)[:, :384]

        if self.whitening_matrix is None:
            return cp.asarray(batch_data, dtype='float32')

        batch_gpu = cp.asarray(batch_data, dtype='float32')
        batch_gpu = gpufilter(batch_gpu, fs=30000, fshigh=300)
        batch_gpu = cp.dot(batch_gpu, self.whitening_matrix)

        return cp.asarray(batch_gpu, dtype='float32')

    @property
    def data(self):
        fake_data = Bunch()
        fake_data.shape = self.shape
        return fake_data
