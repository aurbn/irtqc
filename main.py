from argparse import ArgumentParser

from pyteomics import mzml
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tqdm
import scipy.signal as sgn
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d
from math import log10, ceil, floor
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
        """Returns tuple time,intensity"""
        peaksi, _ = sgn.find_peaks(self.__ints, threshold=self.__ints.mean())
        assert len(peaksi), "No peaks found"
        peaksamp = self.__ints[peaksi]
        maxpeaki = np.argmax(peaksamp)
        apexi = peaksi[maxpeaki]
        return self.__times[apexi], self.__ints[apexi]

    def _get_width_indexs(self, apex_pc):
        _, apexa = self.get_apex()
        t_ = self.__ints>apexa*apex_pc/100
        left = np.where(t_)[0][0]
        right = np.where(t_)[0][-1]
        return left, right

    def get_width_pc(self, apex_pc):
        left, right = self._get_width_indexs(apex_pc)
        return self.__times[left], self.__times[right]

    def get_width_pc_area(self, apex_pc):
        left, right = self._get_width_indexs(apex_pc)
        times = self.__times[left:right]
        times_sh = self.__times[left+1:right+1]
        deltas = times_sh - times
        ints = self.__ints[left:right]
        return np.sum(deltas * ints)

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
    """Single spectrum"""
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

    def _get_apex_around(self, mz, tolerance):
        select = self[mz-tolerance:mz+tolerance]
        peaksi, _ = sgn.find_peaks(select.__inta)
        assert len(peaksi), "No peaks found"
        peaksamp = select.__inta[peaksi]
        maxpeaki = np.argmax(peaksamp)
        apexi = peaksi[maxpeaki]
        return select.__mza[apexi], select.__inta[apexi], apexi

    def get_apex_around(self, mz, tolerance):
        """Returns tuple of apex_mz, apex_int"""
        ap_mz, ap_int, _ = self._get_apex_around(mz, tolerance)
        return ap_mz, ap_int

    def _resample_peak_around(self, mz, tolerance=0.05, ndots = None):
        ndots = ndots if ndots else len(self.__mza)
        select = self[mz-tolerance:mz+tolerance]
        res_x = np.linspace(select.mz[0], select.mz[-1], ndots)
        interp = interp1d(self.__mza, self.__inta)
        res_y = interp(res_x) #resample
        return Spectrum(res_x, res_y, self.__level, self.__prec, self.__time)

    def get_apex_width_pc(self, mz, apex_pc=50, tolerance=0.05):
        """Half width of MS peak"""
        apex_mz, apex_int, index = self._get_apex_around(mz, tolerance)
        resampled = self._resample_peak_around(mz, tolerance)
        t_ = resampled.__inta>(apex_int*apex_pc/100)
        left = np.where(t_)[0][0]
        right = np.where(t_)[0][-1]
        return resampled.__mza[right]-resampled.__mza[left]

    def _get_area(self):
        y1 = self.__mza[:-1]
        y2 = self.__mza[1:]
        x = self.__inta[:-1]
        return np.sum((y2-y1)*x)

    def get_peak_area(self, mz, tolerance=0.05):
        resampled = self._resample_peak_around(mz, tolerance)
        return resampled._get_area()



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
        self._mslevel = mslevel
        self._tic = []
        self._times = []
        self._mza = []
        self._inta = []
        if mslevel == 2:
            self._precs = []
        elif mslevel == 1:
            self._precs = None
        else:
            raise WrongMSLevel(0, 2) #Ewww...

    def finish(self):
        """Should be called after all values are added"""
        self._times = np.array(self._times)
        self._tic = np.array(self._tic)

    def append(self, scan: Scan) -> bool:
        if scan.mslevel == self._mslevel:
            self._tic.append(scan.tic)
            self._times.append(scan.time)
            self._mza.append(scan.mzarray)
            self._inta.append(scan.intarray)
            if scan.mslevel == 2:
                self._precs.append(scan.precursor)
            return True #Appended
        return False #Skip

class MS2Scans(MSnScans):
    def __init__(self):
        self._tolerance = None
        self._tolerance_ppm = None
        super().__init__(2)

    def _mzt(self, mz): #mz transform
        """Rounds mz according to tolerance of tolerance ppm passed to finish"""
        if (not self._tolerance) and (not self._tolerance_ppm):
            return mz
        elif self._tolerance:
            t = int(mz/self._tolerance)*self._tolerance
            return round(t, ceil(abs(log10(t))) + 1) # remove rounding mantissa errors?
        elif self._tolerance_ppm:
            tol = mz*self._tolerance_ppm*1e-6
            t = int(mz/tol)*tol
            return round(t, ceil(abs(log10(t))) + 1) # remove rounding mantissa errors?

    def tic(self, prec):
        """TIC for precursor"""
        prec = self._mzt(prec)
        where = np.where(self._precs == prec)[0]
        return Chromatogram(self._times[where], self._tic[where])

    def finish(self, tolerance=None, tolerance_ppm=None):
        super().finish()
        assert not(tolerance and tolerance_ppm), "Only one tolerance option is acceptable"
        self._tolerance = tolerance
        self._tolerance_ppm = tolerance_ppm
        precs = list(map(self._mzt, self._precs))
        self._allprecs = set(precs)
        self._precs = np.array(list(precs))


class MS1Scans(MSnScans):
    def __init__(self):
        super().__init__(1)

    def __getitem__(self, item):
        if isinstance(item, float):
            index = np.searchsorted(self._times, item)
            return Spectrum(self._mza[index], self._inta[index], self._mslevel,
                            self._precs[index] if self._precs else None,
                            self._times[index])

    def xic(self, mz, ppm):
        if self._mslevel != 1:
            raise WrongMSLevel(1)
        res = []
        for mza, inta in zip(self._mza, self._inta):
            left  = mza.searchsorted(mz * (1 - ppm * 0.5 * 1e-6))
            right = mza.searchsorted(mz * (1 + ppm * 0.5 * 1e-6))
            res.append(inta[left:right].sum())

        res = np.array(res)
        assert len(res) == len(self._times), "Wrong length of chromatogram"
        return Chromatogram(self._times, res)

    @property
    def tic(self):
        return Chromatogram(self._times, self._tic)


class LCMSMSExperiment:
    """Single LCMSMS experiment"""
    def __init__(self, mzmlsource, prec_tolerance = None):
        self.ms1 = MS1Scans()
        self.ms2 = MS2Scans()

        for s_ in mzmlsource:
            scan = Scan(s_)
            self.ms1.append(scan)
            self.ms2.append(scan)

        self.ms1.finish()
        self.ms2.finish(tolerance=prec_tolerance)



if __name__ == '__main__':
    argparser = ArgumentParser(description="iRT peptide QC tool")
    argparser.add_argument('--mzml', type=str, required=True, help="MzML file")
    argparser.add_argument('--targets', type=str, required=True, help="Targets file")
    argparser.add_argument('--ms1-ppm', type=float, default=5, help="MS1 extraction window in ppm")
    argparser.add_argument('--ms2-prec-tolerance', type=float, default=0.01, help="MS2 precursor tolerance")
    argparser.add_argument('--width-1-pc', type=float, default=50, help="Cromatographic width 1 in % of apex")
    argparser.add_argument('--width-2-pc', type=float, default=5, help="Cromatographic width 2 in % of apex")
    argparser = argparser.parse_args()


    ##### FOR TESTING ####
    import pickle
    import time


    #exp = LCMSMSExperiment(tqdm.tqdm(mzml.MzML(argparser.mzml)))

    # mzml_ = list(tqdm.tqdm(mzml.MzML(argparser.mzml)))
    # with open("mzml_.pkl", "wb") as f_:
    #     pickle.dump(mzml_, f_)


    _start_time = time.time()
    with open("mzml_.pkl", "rb") as f_:
        print("Unpickling")
        exp = LCMSMSExperiment(tqdm.tqdm(pickle.load(f_)), prec_tolerance=argparser.ms2_prec_tolerance)
        print(f"Unpickled in {time.time()-_start_time} seconds")
    ### ####

    targets = pd.read_csv(argparser.targets, sep='\t')
    targets_ms1 = targets[["Sequence", "Precursor_Mz"]].drop_duplicates()
    results_ms1 = pd.DataFrame(columns=["Sequence",
                                        "Precursor_Mz",
                                        "Apex_time",
                                        f"Width_{argparser.width_1_pc}_pc_time_start",
                                        f"Width_{argparser.width_1_pc}_pc_time_end",
                                        f"Width_{argparser.width_1_pc}_xic_area",
                                        f"Width_{argparser.width_2_pc}_pc_time_start",
                                        f"Width_{argparser.width_2_pc}_pc_time_end",
                                        f"Width_{argparser.width_2_pc}_xic_area",
                                        f"MS1_mass_apex_mz",
                                        f"MS1_apex_height",
                                        f"MS1_peak_halfwidth",
                                        f"MS1_peak_area",
                                        ])

    #from matplotlib.backends.backend_pdf import PdfPages
    #pdf = PdfPages('MS1.pdf')
    fig, axs = plt.subplots(len(targets_ms1), 2, figsize=(15, 40), gridspec_kw={'width_ratios': [1, 1]})
    plt.subplots_adjust(hspace=0.5)

    n=0

    for k, row in targets_ms1.iterrows():
        mz = row["Precursor_Mz"]
        ch = exp.ms1.xic(mz, argparser.ms1_ppm)
        chs = ch.smooth(sigma=2)
        apext, apexi = ch.get_apex()
        width1 = chs.get_width_pc(argparser.width_1_pc)
        width2 = chs.get_width_pc(argparser.width_2_pc)
        area1 = chs.get_width_pc_area(argparser.width_1_pc)
        area2 = chs.get_width_pc_area(argparser.width_2_pc)
        spec = exp.ms1[apext]
        ms1_apex_mz, ms1_apex_int = spec.get_apex_around(mz, 0.05)
        ms1_hw = spec.get_apex_width_pc(mz, apex_pc=50, tolerance=0.05)
        ms1_area = spec.get_peak_area(mz, tolerance=0.05)

        ### PLOT ###
        axs[n, 0].plot(ch.t, ch.i, "g-")
        #axs[n, 0].plot(xictimes, xic)
        #axs[n, 0].plot(xictimes, asym_peak(xictimes, *popt), 'r-')
        axs[n, 0].vlines(apext, 0, apexi * 1.1)
        axs[n, 0].title.set_text("{}  mz={}  apex@{}".format(n, mz, apext))
        axs[n, 0].set_xlim(15, 30)

        axs[n, 1].plot(ch.t, ch.i, "gx-")
        axs[n, 1].plot(chs.t, chs.i, "rx-")
        #axs[n, 1].plot(xictimes, asym_peak(xictimes, *popt), 'r-')
        axs[n, 1].vlines(apext, 0, apexi)
        axs[n, 1].title.set_text("{}  mz={:.4f}".format(n, mz))
        axs[n, 1].hlines(apexi *0.5, *width1)#, xictimes[right1])
        axs[n, 1].hlines(apexi *0.05, *width2)# xictimes[left2], xictimes[right2])
        axs[n, 1].set_xlim(apext - 0.2, apext + 0.4)
        axs[n, 1].text(0.45, 0.95, f"Area50={area1:.3e}\nArea5  ={area2:.3e}", transform=axs[n, 1].transAxes,
                       fontsize=10, verticalalignment='top')
        n+=1
        ############

        row['Apex_time'] = apext
        row[f"Width_{argparser.width_1_pc}_pc_time_start"] = width1[0]
        row[f"Width_{argparser.width_1_pc}_pc_time_end"] = width1[1]
        row[f"Width_{argparser.width_1_pc}_xic_area"] = area1
        row[f"Width_{argparser.width_2_pc}_pc_time_start"] = width2[0]
        row[f"Width_{argparser.width_2_pc}_pc_time_end"] = width2[1]
        row[f"Width_{argparser.width_2_pc}_xic_area"] = area2
        row[f"MS1_mass_apex_mz"] = ms1_apex_mz
        row[f"MS1_apex_height"] = ms1_apex_int
        row[f"MS1_peak_halfwidth"] = ms1_hw
        row[f"MS1_peak_area"] = ms1_area

        results_ms1 = results_ms1.append(row)

    fig.savefig("MS1.pdf", dpi=1200, format='pdf', bbox_inches='tight')
    results_ms1.to_csv("MS1_test.csv", sep='\t', index=False)

    targets_ms2 = targets[["Sequence", "Precursor_Mz", "Product_Mz"]].drop_duplicates()
    for k, row in targets_ms2.iterrows():
        prec, frag = row["Precursor_Mz"], row["Product_Mz"]
        break

    ex = exp.ms2.tic(prec)
    plt.plot(ex.t, ex.i)

        

