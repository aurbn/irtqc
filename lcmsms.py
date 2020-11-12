import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as sgn
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d
from math import log10, ceil

from scipy.special import erf
from scipy.optimize import curve_fit

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

    def plot(self, *args, ax=None, **kwargs):
        """Plot on MPL axis"""
        if ax:
            ax.plot(self.t, self.i, *args, **kwargs)
        else:
            plt.plot(self.t, self.i, *args, **kwargs)
            plt.show()

    def get_apex(self, threshold=None):
        """Returns tuple (time,intensity)"""
        peaksi, _ = sgn.find_peaks(self.i, threshold=threshold if threshold is None else self.i.mean())
        assert len(peaksi), "No peaks found"
        peaksamp = self.i[peaksi]
        maxpeaki = np.argmax(peaksamp)
        apexi = peaksi[maxpeaki]
        return self.t[apexi], self.i[apexi]

    def _get_width_indexs(self, apex_pc):
        """Returns leftmost and rightmost index where intensity is greater apex_pc% of apex """
        _, apexa = self.get_apex()
        t_ = self.__ints>apexa*apex_pc/100
        left = np.where(t_)[0][0]
        right = np.where(t_)[0][-1]
        return left, right

    def get_width_pc(self, apex_pc):
        """Returns times where intensity raises above  and falls below apex_pc % of apex """
        left, right = self._get_width_indexs(apex_pc)
        return self.t[left], self.t[right]

    def get_width_pc_area(self, apex_pc):
        """Finds area of peake where intensity is greater than apex_pc % of apex"""
        left, right = self._get_width_indexs(apex_pc)
        times = self.t[left:right]
        times_sh = self.t[left+1:right+1]
        deltas = times_sh - times
        ints = self.i[left:right]
        return np.sum(deltas * ints)

    def __getitem__(self, item):
        """Select subchomatogram by time"""
        if isinstance(item, slice):
            if item.step is None:
                left = np.searchsorted(self.t, item.start)
                right = np.searchsorted(self.t, item.stop)
                return Chromatogram(self.t[left:right], self.i[left:right])
            else:
                raise NotImplementedError
        else:
            index = np.searchsorted(self.t, item)
            return self.i[index]

    def smooth(self, kind='gaussian', **kwargs):
        """Smoth gromatogram using filter function"""
        if kind == 'gaussian':
            return Chromatogram(self.t, gaussian_filter1d(self.i, **kwargs))
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
            index = np.searchsorted(self.mz, item)
            return self.i[index]
        elif isinstance(item, slice):
            left = np.searchsorted(self.mz, item.start)
            right = np.searchsorted(self.mz, item.stop)
            return Spectrum(self.mz[left:right], self.i[left:right],
                            self.__level, self.__prec, self.__time)
        else:
            raise TypeError

    def plot(self, *args, ax=None, marks=None, **kwargs):
        if marks:
            mindexs = [np.searchsorted(self.__mza, x) for x in marks]
            mys = [self.__inta[x] for x in mindexs]
        if ax:
            ax.plot(self.__mza, self.__inta, *args, **kwargs)
            if marks:
                ax.scatter(marks, mys, "x")
        else:
            plt.plot(self.__mza, self.__inta, *args, **kwargs)
            if marks:
                plt.scatter(marks, mys, "x")

    def _get_apex_around(self, mz, tolerance):
        """Find apex within mz -/+ tolerance/2"""
        select = self[mz-tolerance/2:mz+tolerance/2]
        peaksi, _ = sgn.find_peaks(select.i)
        assert len(peaksi), "No peaks found"
        peaksamp = select.i[peaksi]
        maxpeaki = np.argmax(peaksamp)
        apexi = peaksi[maxpeaki]
        return select.mz[apexi], select.i[apexi], apexi

    def get_apex_around(self, mz, tolerance):
        """Returns tuple of apex_mz, apex_int"""
        ap_mz, ap_int, _ = self._get_apex_around(mz, tolerance)
        return ap_mz, ap_int

    def _resample_peak_around(self, mz, tolerance=0.05, ndots=None):
        """Get subarea around mz -/+ tolerance and resample it to same amount of measurements"""
        ndots = ndots if ndots else len(self.__mza)
        select = self[mz-tolerance/2:mz+tolerance/2]
        res_x = np.linspace(select.mz[0], select.mz[-1], ndots)
        interp = interp1d(self.mz, self.i)
        res_y = interp(res_x)  # resample
        return Spectrum(res_x, res_y, self.__level, self.__prec, self.__time)

    def get_apex_width_pc(self, mz, apex_pc=50, tolerance=0.05):
        """Half width of MS peak"""
        apex_mz, apex_int, index = self._get_apex_around(mz, tolerance)
        resampled = self._resample_peak_around(mz, tolerance)
        t_ = resampled.i > (apex_int*apex_pc/100)
        left = np.where(t_)[0][0]
        right = np.where(t_)[0][-1]
        return resampled.mz[right]-resampled.mz[left]

    def _get_area(self):
        """Get area under whole spectrum"""
        y1 = self.mz[:-1]
        y2 = self.mz[1:]
        x = self.i[:-1]
        return np.sum((y2-y1)*x)

    def get_peak_area(self, mz, tolerance=0.05):
        """Get area under mz -/+ tolerance/2"""
        resampled = self._resample_peak_around(mz, tolerance)
        return resampled._get_area()


class Scan:
    """Wrapper for quick access to scan fields from Pytomics"""
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
    class _Indexer:
        """Class for indexing internal lists of arrays by lists of indexes"""
        def __init__(self, what):
            self.__what = what

        def __getitem__(self, item):
            if isinstance(item, list) or\
                    (isinstance(item, np.ndarray) and np.issubdtype(item.dtype, np.integer)):
                return list([self.__what[i] for i in item])
            return self.__what[item]

    def __init__(self, mslevel):
        """Constructor for Msn"""
        self._mslevel = mslevel
        self._tic = []    # TIC from spectrometer
        self._times = []  # Time of scans
        self._mza = []    # List of mz arrays
        self._inta = []   # List of intensity arrays
        if mslevel == 2:
            self._precs = []  # Precursors
        elif mslevel == 1:
            self._precs = None

    def finish(self):
        """Should be called after all scans are appended"""
        self._times = np.array(self._times)
        self._tic = np.array(self._tic)

    def append(self, scan: Scan) -> bool:
        """Append scan, if mslevel is different do nothing and return False"""
        if scan.mslevel == self._mslevel:
            self._tic.append(scan.tic)
            self._times.append(scan.time)
            self._mza.append(scan.mzarray)
            self._inta.append(scan.intarray)
            if scan.mslevel == 2:
                self._precs.append(scan.precursor)
            return True  # Appended
        return False  # Skip

    @property
    def mzi(self):
        return self._Indexer(self._mza)

    @property
    def inti(self):
        return self._Indexer(self._inta)


class MS2Scans(MSnScans):
    """MS2 part of experiment"""
    def __init__(self):
        self._tolerance = None
        self._tolerance_ppm = None
        self._allprecs = {}
        super().__init__(2)

    def _mzt(self, mz):  # mz transform
        """Rounds mz according to tolerance or tolerance_ppm passed to finish"""
        if (not self._tolerance) and (not self._tolerance_ppm):
            return mz
        elif self._tolerance:
            t = int(mz/self._tolerance)*self._tolerance
            return round(t, ceil(abs(log10(self._tolerance))) + 1)  # remove rounding mantissa errors?
        elif self._tolerance_ppm:
            tol = mz*self._tolerance_ppm*1e-6
            t = int(mz/tol)*tol
            return round(t, ceil(abs(log10(tol))) + 1)  # remove rounding mantissa errors?

    def extract(self, prec):
        """Extract data for precursor, returns object similar to MS1 subexperiment"""
        prec_ = self._mzt(prec)
        where = np.where(self._precs == prec_)[0]
        return MS2Extracted(prec, self._times[where], self.mzi[where],
                            self.inti[where], self._tic[where])

    def finish(self, tolerance=None, tolerance_ppm=None):
        """Should be called after all scans are added,
         also it's good to specify precursor measurement tolerance"""
        super().finish()
        assert not(tolerance and tolerance_ppm), "Only one tolerance option is acceptable"
        self._tolerance = tolerance
        self._tolerance_ppm = tolerance_ppm
        precs = list(map(self._mzt, self._precs))
        self._allprecs = set(precs)
        self._precs = np.array(list(precs))


class MS1Scans(MSnScans):
    """"MS1 part of experiment"""
    def __init__(self, level=1):
        super().__init__(level)

    def __getitem__(self, item):
        """Get spetrum nearst to time"""
        if isinstance(item, float):
            index = np.searchsorted(self._times, item)
            return Spectrum(self._mza[index], self._inta[index], self._mslevel,
                            self._precs[index] if self._precs else None,
                            self._times[index])

    def xic(self, mz, ppm):
        """Extract Chromatogram of mz with tolerance in ppm"""
        res = []
        for mza, inta in zip(self._mza, self._inta):
            left = mza.searchsorted(mz * (1 - ppm * 0.5 * 1e-6))
            right = mza.searchsorted(mz * (1 + ppm * 0.5 * 1e-6))
            res.append(inta[left:right].sum())

        res = np.array(res)
        assert len(res) == len(self._times), "Wrong length of chromatogram"
        return Chromatogram(self._times, res)

    @property
    def tic(self):
        return Chromatogram(self._times, self._tic)


class MS2Extracted(MS1Scans):
    """Extracted MS2 scans for precursor"""
    def __init__(self, prec, times, mza, inta, tic):
        super().__init__(-2)
        self._mza = mza
        self._inta = inta
        self._times = times
        self._tic = tic
        self._prec = prec
        self._precs = None

    @property
    def prec(self):
        return self._prec

    def __getitem__(self, item):
        if isinstance(item, float):
            return super().__getitem__(item)
        if isinstance(item, slice):
            if item.step is None:
                left = np.searchsorted(self._times, item.start)
                right = np.searchsorted(self._times, item.stop)
                return MS2Extracted(self.prec, self._times[left:right],
                                    self.mzi[left:right], self.inti[left:right],
                                    self._tic[left:right])


class LCMSMSExperiment:
    """Single LCMSMS experiment"""
    def __init__(self, mzmlsource, prec_tolerance=None):
        self.ms1 = MS1Scans()
        self.ms2 = MS2Scans()

        for s_ in mzmlsource:
            scan = Scan(s_)
            self.ms1.append(scan)
            self.ms2.append(scan)

        self.ms1.finish()
        self.ms2.finish(tolerance=prec_tolerance)
