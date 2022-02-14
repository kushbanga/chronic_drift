from pathlib import Path
from math import floor

import numpy as np

from pykilosort import np1_probe, np2_probe, np2_4shank_probe, Bunch
from pykilosort.preprocess import get_whitening_matrix, get_Nbatch
from pykilosort.params import KilosortParams
from pykilosort.learn import extractTemplatesfromSnippets
from pykilosort.datashift2 import standalone_detector, get_drift
from ibllib.dsp.voltage import destripe

from chronic_drift.utils import CustomDataLoader


def detection_pipeline(datasets, probe=np2_4shank_probe(), output_dir=None, **params):
    """
    Runs initial spike detection over a list of datasets and saves the results
    :param datasets: LIst of paths to raw datasets
    :param probe:
    :param output_dir: Where to save results
    :return:
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    params = KilosortParams(scaleproc=1, **params) # avoid rescaling data

    probe.Nchan = 384

    raw_data = CustomDataLoader(datasets, **params.ephys_reader_args)

    whitening_matrix = get_whitening_matrix(raw_data, probe, params)
    n_batches = get_Nbatch(raw_data, params) - 1  # avoid issues with partial last batch

    raw_data = CustomDataLoader(datasets, whitening_matrix=whitening_matrix, **params.ephys_reader_args)

    wTEMP, wPCA = extractTemplatesfromSnippets(
        data_loader=raw_data, probe=probe, params=params, Nbatch=n_batches
    )

    ir = Bunch()
    ir.xc, ir.yc = probe.xc, probe.yc

    # The min and max of the y and x ranges of the channels
    ymin = min(ir.yc)
    ymax = max(ir.yc)
    xmin = min(ir.xc)
    xmax = max(ir.xc)

    # Determine the average vertical spacing between channels.
    # Usually all the vertical spacings are the same, i.e. on Neuropixels probes.
    dmin = np.median(np.diff(np.unique(ir.yc)))
    yup = np.arange(
        start=ymin, step=dmin / 2, stop=ymax + (dmin / 2)
    )  # centers of the upsampled y positions

    # Determine the template spacings along the x dimension
    x_range = xmax - xmin
    npt = floor(
        x_range / 16
    )  # this would come out as 16um for Neuropixels probes, which aligns with the geometry.
    xup = np.linspace(xmin, xmax, npt + 1)  # centers of the upsampled x positions

    spikes = standalone_detector(
        wTEMP, wPCA, params.nPCs, yup, xup, n_batches, raw_data, probe, params
    )

    def save(array, name):
        np.save(output_dir / f'{name}.npy', array)

    save(spikes.times, 'spike_times')
    save(spikes.depths, 'spike_depths')
    save(spikes.amps, 'spike_amps')
    save(raw_data.n_samples, 'dataset_times')

    dshift, yblk = get_drift(spikes, probe, n_batches, params.nblocks, params.genericSpkTh)

    save(dshift, 'dshift')
    save(yblk, 'yblk')


if __name__ == '__main__':
    datasets = [
        Path(r'D:\kilosort-testing\100s_data\hybrid\hybrid_data.bin'),
        Path(r'D:\kilosort-testing\100s_data\hybrid\hybrid_data.bin'),
    ]

    detection_pipeline(datasets, probe=np1_probe(), output_dir='D:/test')
