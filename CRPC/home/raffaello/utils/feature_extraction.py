import numpy as np
def basic_stats(x):
    return {
        'rms': float(np.sqrt(np.mean(x**2))),
        'kurtosis': float(((x - x.mean())**4).mean() / (x.var()**2 + 1e-9)),
        'skew': float(((x - x.mean())**3).mean() / (x.std()**3 + 1e-9)),
    }

