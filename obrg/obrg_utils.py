import math
import numpy as np


def unit_vector(n):
    return n/np.linalg.norm(n)


def ang_div(n1, n2):
    v1_u = unit_vector(n1)
    v2_u = unit_vector(n2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def dist(point, norm, d):
    d = abs(norm[0]*point[0] + norm[1]*point[1] + norm[2] *
            point[2] - d)/math.sqrt(norm[0]**2 + norm[1]**2 + norm[2]**2)
    return d
