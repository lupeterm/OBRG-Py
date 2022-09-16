import os
from typing import List, Set
import numpy as np

from octree import Octree

def get_points(path):
    points = np.loadtxt(path, dtype=float, usecols=(0, 1, 2)).tolist()
    return points

def save_planes(complete_segments: List[Set[Octree]], cloud_path: str):
    folder = cloud_path.rsplit('/', 1)[0]
    results_folder = os.path.join(folder, 'results')
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)
    for i, segment in enumerate(complete_segments):
        filename = os.path.join(results_folder, f'plane_{i}.txt')
        with open(filename,'w') as ofile:
            for leaf in segment:
                for inlier in leaf.indices:
                    ofile.write(f'{leaf.cloud[inlier][0]} {leaf.cloud[inlier][1]} {leaf.cloud[inlier][2]}\n')
        