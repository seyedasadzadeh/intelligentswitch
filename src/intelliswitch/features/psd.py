"""Power spectral density of a vibration window.

Port of the PSD block in `ProBallast_dataGen.m`:
    m    = length(xij);
    nfft = pow2(nextpow2(m-1));
    [pij,fij] = pwelch(xij,[],[],nfft,fs,'onesided','psd');   % Welch PSD
    xdft = fft(xij); ... psdx = abs(xdft).^2/(m*fs); psdx(2:end-1)*=2; % periodogram
then interpolated onto a frequency grid via interp1.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from ..config import Config


def _nfft(m: int) -> int:
    """MATLAB pow2(nextpow2(m-1)): next power of two >= m-1."""
    if m <= 2:
        return 2
    return int(2 ** np.ceil(np.log2(m - 1)))


def psd_welch(x: np.ndarray, fs: int, nfft: int | None = None):
    """Welch one-sided PSD with default Hamming window of length nfft.

    Matches MATLAB `pwelch(x,[],[],nfft,fs,'onesided','psd')`: default window is
    Hamming of length nfft with 50% overlap.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    m = x.size
    nfft = nfft or _nfft(m)
    nperseg = min(nfft, m)
    f, p = welch(
        x, fs=fs, window="hamming", nperseg=nperseg, nfft=nfft,
        noverlap=nperseg // 2, detrend=False, return_onesided=True,
        scaling="density",
    )
    return f, p


def psd_fft(x: np.ndarray, fs: int):
    """One-sided periodogram via FFT (MATLAB psdx block)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    m = x.size
    xdft = np.fft.fft(x)[: m // 2 + 1]
    psdx = (np.abs(xdft) ** 2) / (m * fs)
    psdx[1:-1] *= 2.0
    freq = np.arange(0, fs / 2 + fs / m, fs / m)[: psdx.size]
    return freq, psdx


def psd_on_grid(x: np.ndarray, cfg: Config | None = None) -> np.ndarray:
    """Compute the per-window PSD and interpolate onto `cfg.freq_grid`.

    Applies the dataGen g->m/s^2 scaling and mean removal before transforming.
    Returns the PSD sampled on `cfg.freq_grid` (the feature vector for one group).
    """
    cfg = cfg or Config()
    x = np.asarray(x, dtype=np.float64).ravel()
    x = (x - x.mean()) * cfg.psd_scale_g_to_ms2
    if cfg.psd_method == "fft":
        f, p = psd_fft(x, cfg.fs)
    else:
        f, p = psd_welch(x, cfg.fs)
    return np.interp(cfg.freq_grid, f, p, left=np.nan, right=np.nan)
