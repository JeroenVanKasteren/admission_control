"""
Static functions for the project.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

def conf_int(alpha, data):
    return norm.ppf(1 - alpha / 2) * data.std() / np.sqrt(len(data))


def decay_epsilon(eps, eps_decay):
    """Decay the exploration rate."""
    return eps * eps_decay


def def_sizes(dim):
    """Docstring."""
    sizes = np.zeros(len(dim), np.int32)
    sizes[-1] = 1
    for i in range(len(dim) - 2, -1, -1):
        sizes[i] = sizes[i + 1] * dim[i + 1]
    return sizes


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_time(time_string):
    """Read in time in formats (D)D-HH:MM:SS, (H)H:MM:SS, or (M)M:SS.
    Format in front of time is removed."""
    if ((time_string is not None) & (not pd.isnull(time_string)) &
            (time_string != np.inf)):
        if '): ' in time_string:  # if readable format
            time_string = time_string.split('): ')[1]
        if '-' in time_string:
            days, time = time_string.split('-')
        elif time_string.count(':') == 1:
            days, time = 0, '0:' + time_string
        else:
            days, time = 0, time_string
        hour, minutes, sec = [int(x) for x in time.split(':')]
        return (((int(days) * 24 + hour) * 60 + minutes) * 60 + sec - 60)
    else:
        return np.Inf


def round_significance(x, digits=1):
    return 0 if x == 0 else np.round(x, -int(np.floor(np.log10(abs(x)))) -
                                     (-digits + 1))


def sec_to_time(time):
    """Convert seconds to minutes and return readable format."""
    time = int(time)
    if time >= 60 * 60:
        return (f"(HH:MM:SS): {time // (60 * 60):02d}:{(time // 60) % 60:02d}:"
                f"{time % 60:02d}")
    else:
        return f"(MM:SS): {time // 60:02d}:{time % 60:02d}"


def strip_split(x):
    if ',' in x:
        return np.array([float(i) for i in x.strip('[]').split(', ')])
    else:
        return np.array([float(i) for i in x.strip('[]').split()])


def time_print(self, time):
    """Convert seconds to readable format."""
    print(f'Time: {time / 60:.0f}:{time - 60 * int(time / 60):.0f} min.\n')


def update_mean(mean, x, n):
    """Welford's method to update the mean. Can be set to numba function."""
    # avg_{n-1} = avg_{n-1} + (x_n - avg_{n-1})/n
    return mean + (x - mean) / n
