from argparse import ArgumentParser

from pyteomics import mzml
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tqdm
import scipy.signal as sgn
from scipy.ndimage.filters import gaussian_filter1d
import itertools

from scipy.special import erf
from scipy.optimize import curve_fit

import sys

# def precmz_to_float(mz, listfloat):
#     for f in listfloat:
#         if me(mz, f, eic_ppm):
#             return f
#     assert False, "No mass in list"

class WrongMSLevel(Exception):
    """Wrong MS level"""

    def __init__(self, expected, given = None):
        """Constructor for WrongMSLevel"""
        self.__given = given
        self.__expected = expected

    def __str__(self):
        if self.__given:
            if isinstance(self.__given, int):
                return f'Expected MS level {self.__expected}, given {self.__given}'
            elif isinstance(self.__given, Scan):
                return f'Expected MS level {self.__expected}, given {self.__given.mslevel}'
            else:
                raise TypeError
        else:
            return f'Wrong MS level, should be {self.__expected}'

class T:
    def __init__(self, time):
        self.__time  = time

    def __str__(self):
        return f'{self.__time} min'

    def __int__(self):
        return self.__time

class Chromatogram:
    """Simple chromatogram"""

    def __init__(self, times, ints):
        """Constructor for Chromatogram"""
        self.__times = times
        self.__ints = ints

    @property
    def t(self):
        """Returns internal times array"""
        return self.__times

    @property
    def i(self):
        """Returns internal intensity array"""
        return self.__ints

    def plot(self, ax, *args, **kwargs):
        ax.plot(self.__times, self.__ints, *args, **kwargs)

    def get_apex(self):
        peaksi, _ = sgn.find_peaks(self.__ints, threshold=self.__ints.mean())
        assert len(peaksi), "No peaks found"
        peaksamp = self.__ints[peaksi]
        maxpeaki = np.argmax(peaksamp)
        apexi = peaksi[maxpeaki]
        return apexi, self.__ints[apexi], self.__times[apexi]

    def get_apex_time(self):
        return self.get_apex()[2]

    def get_apex_int(self):
        return self.get_apex()[1]

    def _get_width_indexs(self, apex_pc):
        apexa = self.get_apex_int()
        left = np.where(self.__ints>apexa*apex_pc/100)[0][0]
        right = np.where(self.__ints>apexa*apex_pc/100)[0][-1]
        return left, right

    def get_width(self, apex_pc):
        left, right = self._get_width_indexs(apex_pc)
        return self.__times[left], self.__times[right]

    def get_width_pc_area(self, apex_pc):
        left, right = self._get_width_indexs(apex_pc)
        times = self.__times[left:right]
        times_sh = self.__times[left+1:right+1]
        deltas = times_sh - times
        ints = self.__ints[left:right]
        return sum(deltas * ints)

    def __getitem__(self, item):
        if isinstance(item, slice):
            if item.step is None:
                left = np.searchsorted(self.__times, item.start)
                right = np.searchsorted(self.__times, item.stop)
                return Chromatogram(self.__times[left:right], self.__ints[left:right])
            else:
                raise NotImplementedError
        else:
            index = np.searchsorted(self.__times, item)
            return self.__ints[index]

    def smooth(self, type='gaussian', **kwargs):
        if type == 'gaussian':
            return Chromatogram(self.__times, gaussian_filter1d(self.__ints, **kwargs))
        else:
            raise NotImplementedError('Only gaussian smoothing is supported')


class Spectrum:
    """Simgle spectrum"""

    def __init__(self, mza, inta, mslevel=None, precursor=None, time=None):
        """Constructor for Spectrum"""

        self.__mza = mza
        self.__inta = inta
        self.__level = mslevel
        self.__prec = precursor
        self.__time = time

    @property
    def mz(self):
        return self.__mza

    @property
    def i(self):
        return self.__inta

    def __getitem__(self, item):
        if isinstance(item, float):
            index = np.searchsorted(self.__mza, item)
            return self.__inta[index]
        elif isinstance(item, slice):
            left = np.searchsorted(self.__mza, item.start)
            right = np.searchsorted(self.__mza, item.stop)
            return Spectrum(self.__mza[left:right], self.__inta[left:right],
                            self.__level, self.__prec, self.__time)
        else:
            raise TypeError


class Scan:
    """Wrapper for quick access to scan fields"""
    def __init__(self, scan):
        self.__s = scan

    @property
    def time(self) -> float:
        return self.__s['scanList']['scan'][0]['scan start time']

    @property
    def tic(self) -> float:
        return self.__s['total ion current']

    @property
    def mzarray(self) -> np.ndarray:
        return self.__s['m/z array']

    @property
    def intarray(self) -> np.ndarray:
        return self.__s['intensity array']

    @property
    def mslevel(self) -> int:
        return self.__s['ms level']

    @property
    def precursor(self) -> float:
        if self.mslevel == 2:
            return self.__s['precursorList']['precursor'][0]\
                ['selectedIonList']['selectedIon'][0]['selected ion m/z']
        else:
            raise WrongMSLevel(2, self)


class MSnScans:
    """Class for storing MSn data for single MS level"""

    def __init__(self, mslevel):
        """Constructor for Msn"""
        self.__mslevel = mslevel
        self.__tic = []
        self.__times = []
        self.__mza = []
        self.__inta = []
        if mslevel == 2:
            self.__precs = []
        elif mslevel == 1:
            self.__precs = None
        else:
            raise WrongMSLevel(0, 2) #Ewww...

    def append(self, scan: Scan) -> bool:
        if scan.mslevel == self.__mslevel:
            self.__tic.append(scan.tic)
            self.__times.append(scan.time)
            self.__mza.append(scan.mzarray)
            self.__inta.append(scan.intarray)
            if scan.mslevel == 2:
                self.__precs.append(scan.precursor)
            return True #Appended
        return False #Skip

    def finish(self):
        """Should be called after all values are added"""

        self.__times = np.array(self.__times)
        self.__tic = np.array(self.__tic)
        if self.__mslevel == 2:
            self.__precs = np.array(self.__precs)

    @property
    def tic(self):
        return Chromatogram(self.__times, self.__tic)

    def xic(self, mz, ppm):
        if self.__mslevel != 1:
            raise WrongMSLevel(1)
        res = []
        for mza, inta in zip(self.__mza, self.__inta):
            left = mza.searchsorted(mz * (1 - ppm * 1e-6))
            right = mza.searchsorted(mz * (1 + ppm * 1e-6))
            res.append(inta[left:right].sum())

        res = np.array(res)
        assert len(res) == len(self.__times), "Wrong length of chromatogram"
        return Chromatogram(self.__times, res)

    def __getitem__(self, item):
        if isinstance(item, float):
            index = np.searchsorted(self.__times, item)
            return Spectrum(self.__mza[index], self.__inta[index], self.__mslevel,
                            self.__precs[index] if self.__precs else None,
                            self.__times[index])
        else:
            raise NotImplementedError


class LCMSMSExperiment:
    """Class for single LCMSMS experiment"""

    def __init__(self, mzmlsource):
        self.ms1 = MSnScans(mslevel=1)
        self.ms2 = MSnScans(mslevel=2)

        for s_ in mzmlsource:
            scan = Scan(s_)
            self.ms1.append(scan)
            self.ms2.append(scan)

        self.ms1.finish()
        self.ms2.finish()



if __name__ == '__main__':
    argparser = ArgumentParser(description="iRT peptide QC tool")
    argparser.add_argument('--mzml', type=str, required=True, help="MzML file")
    argparser.add_argument('--targets', type=str, required=True, help="Targets file")
    argparser.add_argument('--ms1-ppm', type=float, default=0.05, help="MS1 extraction window in ppm")
    argparser.add_argument('--width-1-pc', type=float, default=50, help="Cromatographic width 1 in % of apex")
    argparser.add_argument('--width-2-pc', type=float, default=5, help="Cromatographic width 2 in % of apex")
    argparser = argparser.parse_args()


    exp = LCMSMSExperiment(tqdm.tqdm(mzml.MzML(argparser.mzml)))

    fig = plt.figure()
    ax = fig.add_subplot()
    exp.ms1.tic.plot(ax, "Ms1", "g-")
    fig.savefig("tic.png")

    targets = pd.read_csv("./iRT2/qc1_targets.csv", sep='\t')

    pass



