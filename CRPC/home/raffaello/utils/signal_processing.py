import numpy as np
def spectral_centroid(x, fs):
    X = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1/fs)
    return float((freqs * X).sum() / (X.sum() + 1e-9))

