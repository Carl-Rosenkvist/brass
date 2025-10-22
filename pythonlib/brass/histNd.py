import numpy as np


class HistND:
    """
    Fast N-D histogram with bincount + flat indexing.

    Parameters
    ----------
    edges : sequence of 1D arrays
        Bin edges per axis, length D. Uses left-closed, right-open bins [e[i], e[i+1]).
    dtype : numpy dtype, default float
        Storage dtype for counts.
    track_variance : bool, default False
        If True, also track sum of weights squared (sumw2) for error estimation.
    """

    def __init__(self, edges, dtype=float, track_variance=False):
        self.edges = [np.asarray(e) for e in edges]
        self.D = len(self.edges)
        if self.D == 0:
            raise ValueError("Need at least 1 dimension.")
        self.nbins = np.array([len(e) - 1 for e in self.edges], dtype=np.int64)
        if np.any(self.nbins <= 0):
            raise ValueError("Each edges array must have at least 2 entries.")

        self.size = int(np.prod(self.nbins))
        self.dtype = np.dtype(dtype)

        # Storage
        self.counts = np.zeros(self.nbins, dtype=self.dtype)
        self.track_variance = bool(track_variance)
        if self.track_variance:
            self.sumw2 = np.zeros(self.nbins, dtype=float)

        # Per-axis uniform detection and params
        self._uniform = []
        self._mins = []
        self._scales = []
        for e, n in zip(self.edges, self.nbins):
            d = np.diff(e)
            uni = np.allclose(d, d[0])
            self._uniform.append(uni)
            if uni:
                self._mins.append(e[0])
                self._scales.append(n / (e[-1] - e[0]))
            else:
                self._mins.append(None)
                self._scales.append(None)

        # C-order strides for flattening indices: flat = sum(b_k * stride_k)
        self._strides = np.empty(self.D, dtype=np.int64)
        acc = 1
        for k in range(self.D - 1, -1, -1):
            self._strides[k] = acc
            acc *= self.nbins[k]

    # ---------- arithmetic ----------
    def __add__(self, other):
        """Return a new histogram that is the elementwise sum of self and other."""
        if not isinstance(other, HistND):
            return NotImplemented
        self._check_compat(other)

        out_track = self.track_variance or other.track_variance
        out_dtype = np.result_type(self.counts.dtype, other.counts.dtype)

        out = HistND(self.edges, dtype=out_dtype, track_variance=out_track)
        out.counts = self.counts.astype(out_dtype, copy=False) + other.counts.astype(
            out_dtype, copy=False
        )

        if out_track:
            a = getattr(self, "sumw2", None)
            b = getattr(other, "sumw2", None)
            if a is None and b is None:
                out.sumw2 = np.zeros_like(out.counts, dtype=float)
            else:
                if a is None:
                    a = np.zeros_like(out.counts, dtype=float)
                if b is None:
                    b = np.zeros_like(out.counts, dtype=float)
                out.sumw2 = a + b
        return out

    def __iadd__(self, other):
        """In-place elementwise sum (self += other). May upcast dtype and enable variance."""
        if not isinstance(other, HistND):
            return NotImplemented
        self._check_compat(other)

        # dtype promotion
        out_dtype = np.result_type(self.counts.dtype, other.counts.dtype)
        if self.counts.dtype != out_dtype:
            self.counts = self.counts.astype(out_dtype, copy=True)
            self.dtype = np.dtype(out_dtype)

        self.counts += other.counts.astype(out_dtype, copy=False)

        # enable / add sumw2 if RHS has it
        if getattr(other, "sumw2", None) is not None:
            if not self.track_variance:
                self.sumw2 = np.zeros_like(self.counts, dtype=float)
                self.track_variance = True
            self.sumw2 += other.sumw2
        return self

    def __radd__(self, other):
        # supports sum([hist1, hist2, ...], start=0) and plain sum(list_of_hists)
        if other == 0:
            return self
        return NotImplemented

    # ---------- filling ----------
    @staticmethod
    def _bin_irregular(edges, vals):
        # [e[i], e[i+1]), right-open (except last edge excluded)
        return np.searchsorted(edges, vals, side="right") - 1

    def _bin_uniform(self, k, vals):
        return np.floor((vals - self._mins[k]) * self._scales[k]).astype(np.int64)

    def _bin_axes(self, coords):
        bins = []
        for k, (e, uni, v) in enumerate(zip(self.edges, self._uniform, coords)):
            b = self._bin_uniform(k, v) if uni else self._bin_irregular(e, v)
            bins.append(b)
        return bins

    def _accumulate(self, bins, weights=None):
        # in-range mask across all axes
        ok = np.ones_like(bins[0], dtype=bool)
        for k, b in enumerate(bins):
            ok &= (b >= 0) & (b < self.nbins[k])
        if not np.any(ok):
            return

        b_ok = [b[ok] for b in bins]

        # make scalar weights broadcastable
        if weights is None:
            w_ok = None
        else:
            w = np.asarray(weights)
            if w.ndim == 0:
                w_ok = np.full(b_ok[0].shape, float(w))
            else:
                w_ok = w[ok]

        # flatten N-D indices
        flat = np.zeros_like(b_ok[0], dtype=np.int64)
        for k in range(self.D):
            flat += b_ok[k] * self._strides[k]

        # accumulate via bincount
        if w_ok is None:
            add = np.bincount(flat, minlength=self.size)
            if self.track_variance:
                add2 = add  # since w=1 => w^2=1
        else:
            add = np.bincount(flat, weights=w_ok, minlength=self.size)
            if self.track_variance:
                add2 = np.bincount(flat, weights=w_ok * w_ok, minlength=self.size)

        add = add.reshape(self.nbins)
        self.counts += add.astype(self.counts.dtype, copy=False)
        if self.track_variance:
            self.sumw2 += add2.reshape(self.nbins)

    def fill(self, *coords, mask=None, weights=None):
        """
        Fill with raw coordinates (one array or scalar per axis).
        """
        if len(coords) != self.D:
            raise ValueError(f"Expected {self.D} coordinate arrays.")
        arrs = [np.asarray(c) for c in coords]
        arrs = [a if a.ndim > 0 else a[None] for a in arrs]
        if mask is not None:
            m = np.asarray(mask)
            arrs = [a[m] for a in arrs]
            if weights is not None:
                w = np.asarray(weights)
                weights = w[m] if w.ndim > 0 else w
        bins = self._bin_axes(arrs)
        self._accumulate(bins, weights)

    def fill_binned(self, *bins, mask=None, weights=None):
        """
        Fill with precomputed integer bin indices per axis.
        """
        if len(bins) != self.D:
            raise ValueError(f"Expected {self.D} bin index arrays.")
        b = [np.asarray(bi) for bi in bins]
        if mask is not None:
            m = np.asarray(mask)
            b = [bi[m] for bi in b]
            if weights is not None:
                w = np.asarray(weights)
                weights = w[m] if w.ndim > 0 else w
        self._accumulate(b, weights)

    # ---------- utilities ----------
    def merge_(self, other):
        """In-place merge (same as +=), kept for API compatibility."""
        return self.__iadd__(other)

    def normalized_copy(self, norm):
        """Return a new histogram with counts divided by `norm` (sumw2 by norm^2)."""
        out = HistND(self.edges, dtype=self.dtype, track_variance=self.track_variance)
        out.counts = self.counts / float(norm)
        if self.track_variance:
            out.sumw2 = self.sumw2 / (float(norm) ** 2)
        return out

    def errors(self):
        """Standard deviation per bin assuming uncorrelated weights: sqrt(sumw2)."""
        if not self.track_variance:
            raise RuntimeError("Enable track_variance=True to get errors().")
        return np.sqrt(self.sumw2)

    def to_numpy(self):
        """Return (counts copy, edges copy list) similar to numpy.histogramdd outputs."""
        return self.counts.copy(), [e.copy() for e in self.edges]

    def project(self, axes_to_keep):
        """
        Sum over all axes not in axes_to_keep.
        Returns (projected_counts, kept_edges).
        """
        axes_to_keep = tuple(sorted(axes_to_keep))
        axes_all = tuple(range(self.D))
        axes_sum = tuple(i for i in axes_all if i not in axes_to_keep)
        counts_proj = self.counts.sum(axis=axes_sum)
        edges_kept = [self.edges[i] for i in axes_to_keep]
        return counts_proj, edges_kept

    def density(self):
        """
        Counts per unit hyper-volume (divide by product of bin widths).
        """
        vol = np.ones(self.nbins, dtype=float)
        for ax, e in enumerate(self.edges):
            d = np.diff(e)
            shape = [1] * self.D
            shape[ax] = self.nbins[ax]
            vol *= d.reshape(shape)
        return self.counts / vol

    def copy(self):
        """Deep copy of the histogram structure and contents."""
        out = HistND(self.edges, dtype=self.dtype, track_variance=self.track_variance)
        out.counts = self.counts.copy()
        if self.track_variance:
            out.sumw2 = self.sumw2.copy()
        return out

    def astype(self, dtype):
        """Return a new histogram with counts cast to `dtype` (sumw2 stays float)."""
        out = HistND(self.edges, dtype=dtype, track_variance=self.track_variance)
        out.counts = self.counts.astype(dtype, copy=True)
        if self.track_variance:
            out.sumw2 = self.sumw2.copy()
        return out

    def _check_compat(self, other):
        if self.nbins.tolist() != other.nbins.tolist() or any(
            not np.array_equal(a, b) for a, b in zip(self.edges, other.edges)
        ):
            raise ValueError("Histogram shapes/edges must match.")

    def __repr__(self):
        return (
            f"HistND(nbins={tuple(self.nbins.tolist())}, "
            f"dtype={self.counts.dtype}, track_variance={self.track_variance})"
        )
