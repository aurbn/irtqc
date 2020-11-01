from argparse import ArgumentParser

from pyteomics import mzml, mgf, tandem
from functools import partial
import numpy as np
import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt
from pprint import pprint
import tqdm
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



# def get_mz_intensity(scan, mz, ppm):
#     mza, inta = get_spectrum(scan)
#     left = mza.searchsorted(mz * (1 - eic_ppm * 1e-6))
#     right = mza.searchsorted(mz * (1 + eic_ppm * 1e-6))
#     return inta[left:right].sum()


def get_xic_apex(xic, times=None):
    peaksi = sgn.find_peaks(xic, threshold=np.mean(xic))[0]
    peaksamp = xic[peaksi]
    maxpeaki = np.argmax(peaksamp)
    apexi = peaksi[maxpeaki]
    # print(peaksi)
    return apexi, xic[apexi], times[apexi] if len(times) else None


def dict_npize(d):
    return {k: np.array(v) for k, v in d.items()}


def precmz_to_float(mz, listfloat):
    for f in listfloat:
        if me(mz, f, eic_ppm):
            return f
    assert False, "No mass in list"

class WrongMSLevel(Exception):
    """Wrong MS level"""

    def __init__(self, expected, given = None):
        """Constructor for WrongMSLevel"""
        self.__given = given
        self.__expected = expected

    def __str__(self):
        if self.__given:
            if type(self) == int:
                return f'Expected MS level {self.__expected}, given {self.__given}'
            elif type(self.__given) == Scan:
                return f'Expected MS level {self.__expected}, given {self.__given.mslevel}'
            else:
                raise ValueError
        else:
            return f'Wrong MS level, should be {self.__expected}'


class Scan:
    """Wrapper for quick access to scan fields"""
    def __init__(self, scan):
        self.__s = scan

    @property
    def time(self):
        return self.__s['scanList']['scan'][0]['scan start time']

    @property
    def tic(self):
        return self.__s['total ion current']

    @property
    def mzarray(self):
        return self.__s['m/z array']

    @property
    def intarray(self):
        return self.__s['intensity array']

    @property
    def mslevel(self):
        return self.__s['ms level']

    @property
    def precursor(self):
        if self.mslevel == 2:
            return self.__s['precursorList']['precursor'][0]\
                ['selectedIonList']['selectedIon'][0]['selected ion m/z']
        else:
            raise WrongMSLevel(2, self)


class Tic:
    """Class for TIC chromatogram"""

    def __init__(self,):
        self.__times = []
        self.__ints = []

    def append(self, scan: Scan):
        self.__times.append(scan.time)
        self.__ints.append(scan.tic)


class MSn:
    """Class for storing MSn data for single MS level"""

    def __init__(self, mslevel):
        """Constructor for Msn"""
        self.__mslevel = mslevel
        self.__times =[]
        self.__mza = []
        self.__inta = []
        if mslevel == 2:
            self.__precs = []
        else:
            self.__precs = None

    def append(self, scan: Scan) -> bool:
        if scan.mslevel == self.__mslevel:
            self.__times.append(scan.time)
            self.__mza.append(scan.mzarray)
            self.__inta.append(scan.intarray)
            if scan.mslevel == 2:
                self.__precs.append(scan.precursor)
            return True #Appended
        return False #Skip


class MsExperiment:
    """Class for single MSMS experiment"""

    def __init__(self, mzmlsource):
        self.tic = Tic()
        self.ms1 = MSn(mslevel=1)
        self.ms2 = MSn(mslevel=2)

        for s_ in mzmlsource:
            scan = Scan(s_)
            self.tic.append(scan)
            self.ms1.append(scan)
            self.ms2.append(scan)


            #     for t in targets_list:
            #         xics[t].append(get_mz_intensity(scan, t, eic_ppm))
            #
            # if scan['ms level'] == 2:
            #     # import ipdb; ipdb.set_trace() # BREAKPOINT
            #     prec = get_prec_mz(scan)
            #     prec = precmz_to_float(prec, targets_frags.keys())
            #     for frag in targets_frags[prec]:
            #         ms2_xics[(prec, frag)].append(get_mz_intensity(scan, frag, ms2_window_ppm))
            #         ms2_xictimes[(prec, frag)].append(scantime(scan))
            #
            #     #  ms2_scans_mz[(prec,frag)].append(scan['m/z array'])
            #     #  ms2_scans_int[(prec,frag)].append(scan['intensity array'])
            #     #  ms2_scans_time[(prec,frag)].append(scantime(scan))


if __name__ == '__main__':
    argparser = ArgumentParser(description="iRT peptide QC tool")
    argparser.add_argument('--mzml', type=str, required=True, help="MzML file")
    argparser.add_argument('--targets', type=str, required=True, help="Targets file")
    argparser.add_argument('--ms1-ppm', type=float, default=0.05, help="MS1 extraction window in ppm")
    argparser.add_argument('--width-1-pc', type=float, default=50, help="Cromatographic width 1 in % of apex")
    argparser.add_argument('--width-2-pc', type=float, default=5, help="Cromatographic width 2 in % of apex")
    argparser = argparser.parse_args()


    numscans = 0
    exp = MsExperiment(tqdm.tqdm(mzml.MzML(argparser.mzml)))




