import math
import numpy as np
from typing import Callable
import cProfile
import io
import pstats
from pstats import SortKey


def unit_vector(v):
    return v/math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])


def ang_div(n1, n2):
    v1_u = unit_vector(n1)
    v2_u = unit_vector(n2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def dist(point, norm, d):
    d = abs(norm[0]*point[0] + norm[1]*point[1] + norm[2] *
            point[2] - d)/math.sqrt(norm[0]**2 + norm[1]**2 + norm[2]**2)
    return d


# Benchmarking Decorator
def bench_this(func_to_profile: Callable):
    """
    Wraps cProfile benchmarking around given input Callable func_to_profile.
    Returns whatever func_to_profile would return.

    >>> @bench_this
        def func():
            ...

    """
    def wrapper(self, *args):
        pr = cProfile.Profile()
        pr.enable()
        ret_val = func_to_profile(self, *args)
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return ret_val
    return wrapper
