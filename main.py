from pyteomics import mzml, mgf, tandem
from functools import partial
import numpy as np
import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt
from pprint import pprint
from tqdm import tqdm_notebook as tqdm
import scipy.signal as sgn
from scipy.ndimage.filters import gaussian_filter1d
import bokeh.plotting as bk
import itertools

from scipy.special import erf
from scipy.optimize import curve_fit

import sys
#import warnings; warnings.simplefilter('ignore')

mdelta = 1e-4
ms2_window_ppm = 0.001*1e6
eic_ppm = 2.5



#eic_width_unit	ppm
eic_width_value = 5.0
width_1_pc = 0.50
width_2_pc = 0.05



ms1_peak_apexs = []
ms1_xic_ints =[]
ms1_fit_area = []
ms1_fit_width = []

#time boundaries
ms1_pc_w1_left = []
ms1_pc_w1_right = []
ms1_pc_w2_left = []
ms1_pc_w2_right = []

#index boundaries
ms1_pc_w1_left_i = []
ms1_pc_w1_right_i = []
ms1_pc_w2_left_i = []
ms1_pc_w2_right_i = []

ms1_xic_area_w1 = []
ms1_xic_area_w2 = []


def asym_peak(t, *pars):
    #'from Anal. Chem. 1994, 66, 1294-1301'
    a0 = pars[0]  # peak area
    a1 = pars[1]  # elution time
    a2 = pars[2]  # width of gaussian
    a3 = pars[3]  # exponential damping term
    f = (a0 / 2 / a3 * np.exp(a2 ** 2 / 2.0 / a3 ** 2 + (a1 - t) / a3)
         * (erf((t - a1) / (np.sqrt(2.0) * a2) - a2 / np.sqrt(2.0) / a3) + 1.0))
    return f


def mq(m1, m2, delta):
    return abs(m1 - m2) < delta


#def mw(m1, m2):
#    return abs(m1 - m2) < mwindow


def me(m1, m2, ppm):
    return abs(m1 - m2) < eic_ppm * 1e-6 * (m1 + m2) / 2


def get_prec_mz(scan):
    assert scan['ms level'] == 2, "Ms level must be == 2"
    return scan['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['selected ion m/z']


def get_spectrum(scan):
    return scan['m/z array'], scan['intensity array']


def get_mz_intensity(scan, mz, ppm):
    mza, inta = get_spectrum(scan)
    left = mza.searchsorted(mz * (1 - eic_ppm * 1e-6))
    right = mza.searchsorted(mz * (1 + eic_ppm * 1e-6))
    return inta[left:right].sum()


def get_xic_apex(xic, times=None):
    peaksi = sgn.find_peaks(xic, threshold=np.mean(xic))[0]
    peaksamp = xic[peaksi]
    maxpeaki = np.argmax(peaksamp)
    apexi = peaksi[maxpeaki]
    # print(peaksi)
    return apexi, xic[apexi], times[apexi] if len(times) else None


def scantime(scan):
    return scan['scanList']['scan'][0]['scan start time']


def dict_npize(d):
    return {k: np.array(v) for k, v in d.items()}


def precmz_to_float(mz, listfloat):
    for f in listfloat:
        if me(mz, f, eic_ppm):
            return f
    assert False, "No mass in list"



if __name__ == '__main__':
   pass
