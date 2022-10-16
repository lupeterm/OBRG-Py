from time import time
from typing import Dict, List, Set
import open3d as o3d
import numpy as np
from collections import deque
from .obrg_io import get_points, save_planes, save_time
from .obrg_utils import ang_div, dist
from .octree import Octree, get_neighbor_count_same_cluster
from .visualization import draw_complete, draw_incomplete, draw_leaf_centers
from tqdm import tqdm

# THRESHOLD PARAMETERS USED IN OBRG
RES_TH = 0.08   # in meter, i guess?
D_TH = 0.08     # in meter, i guess?
ANG_TH = 0.3    # no idea, works somewhat fine
MIN_SEGMENT = 5000  # min points of segment


def obrg(O: Octree) -> List[Set[Octree]]:
    R: List[Set[Octree]] = list()
    a = O.leaves
    a.sort(key=lambda x: x.residual)
    A = deque(a)
    visited = dict()
    while len(A) > 0:
        R_c: Set[Octree] = set()
        S_c: Set[Octree] = set()
        v_min = A.popleft()
        if v_min.residual > RES_TH:
            break
        S_c.add(v_min)
        R_c.add(v_min)
        while len(S_c) > 0:
            v_i = S_c.pop()
            B_c = v_i.get_neighbors()
            for v_j in B_c:
                ang = ang_div(v_i.normal, v_j.normal)
                if v_j in A and ang <= ANG_TH:
                    # check if already inserted somewhere
                    R_c.add(v_j)
                    A.remove(v_j)
                    if v_j.residual < RES_TH:
                        S_c.add(v_j)
            m = sum([len(l.indices) for l in R_c])
            if m > MIN_SEGMENT:
                inclu = None
                for l in R_c:
                    if l in visited.keys():
                        inclu = visited[l]
                if inclu is not None and len(R) > 0:
                    for l in R:
                        if inclu in l:
                            for l2 in R_c:
                                l.add(l2)
                            break
                else:
                    for l in R_c:
                        visited[l] = v_i
                    R.append(R_c)
            else:
                for l in R_c:
                    l.is_unallocated = True
    return sorted(R, key=lambda x: len(x), reverse=True)


def extract_boundary_voxels(cluster: Set[Octree]) -> Set[Octree]:
    cluster_centers = [tuple(voxel.center)
                       for voxel in cluster]
    boundaries = set([leaf for leaf in cluster if get_neighbor_count_same_cluster(
        leaf, cluster_centers)])
    return boundaries


def check_planarity(r_i: Set[Octree]) -> bool:
    avg_norm = np.mean([l.normal for l in r_i], axis=0)
    avg_d = np.mean([l.d for l in r_i], axis=0)
    num_points = sum([len(l.indices) for l in r_i])
    planar = 0
    ds = []
    for leaf in r_i:
        for index in leaf.indices:
            d = dist(leaf.cloud[index], avg_norm, avg_d)
            ds.append(d)
            if d < D_TH:
                planar += 1
    return (planar / num_points) > 0.8


def fast_refine(O: Octree, R_i: List[Octree], V_b: Set[Octree]) -> None:
    if len(V_b) == 0:
        return
    S = V_b.copy()
    norm_R_i = sum([l.normal for l in R_i]) / len(R_i)
    d_R_i = sum([l.d for l in R_i]) / len(R_i)
    to_be_added: Set[int] = set()
    visited = set()
    while len(S) > 0:
        v_j = S.pop()
        visited.add(v_j)
        B = v_j.get_neighbors()
        for v_k in B:
            if v_k.is_unallocated:
                for index in v_k.indices:
                    if dist(v_k.cloud[index], norm_R_i, d_R_i) < D_TH:
                        to_be_added.add(index)
                        if v_k not in visited:
                            S.add(v_k)
    tmp = V_b.pop()
    for index in to_be_added:
        tmp.indices.append(index)
    V_b.add(tmp)


def general_refinement(O: Octree, R_i: List[Octree], b_v: List[Octree], kdtree) -> None:
    S: Set[Octree] = set(b_v)
    # visited = set()
    to_add: Dict[Octree, Set[int]] = {v: set() for v in b_v}
    while len(S) > 0:
        v_j = S.pop()
        nb_points: Dict[Octree, List] = v_j.get_buffer_zone_points(kdtree)
        for neighbor, nb_indices in nb_points.items():
            for nbi in nb_indices:
                a = ang_div(v_j.normal, neighbor.normals[nbi])
                b = dist(neighbor.cloud[nbi], v_j.normal, v_j.d)
                if a <= ANG_TH and b < RES_TH:
                    to_add[v_j].add(nbi)
    for k, v in to_add.items():
        for val in v:
            k.indices.append(val)


def refinement(is_planar, oc, incomplete_segment, b_v, kdtree):
    if is_planar:
        # fast refinement
        print('planar!')
        fast_refine(oc, incomplete_segment, b_v)
    else:
        print('not planar')
        # general refinement for non planar segments
        general_refinement(oc, incomplete_segment, b_v, kdtree)


def calculate(cloud_path: str, output_path: str, debug=False):
    # Preparation:
    # read point cloud
    points = get_points(cloud_path)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    bb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(points))
    KDTree = o3d.geometry.KDTreeFlann(cloud)

    ####  PHASE A ####
    print('Entering Phase A')
    start = time()
    # A1a voxelization
    norms = np.asarray(cloud.normals)
    oc = Octree(points, center=bb.get_center(),
                normals=np.asarray(cloud.normals))
    oc.create(bb.get_max_extent())
    # A1b saliency feature estimation

    for leaf in oc.leaves:
        if len(leaf.indices) > 0:
            leaf.calc_n_r()
    pre = time()-start

    # A2 voxel based Region Growing
    print('Entering OBRG')
    start = time()
    incomplete_segments = obrg(oc)
    elapsed = time()-start
    print(f'time spent in obrg: {elapsed} seconds')
    if debug:
        np.random.seed(0)
        colors = [np.random.rand(3) for _ in range(len(incomplete_segments))]

    #### PHASE B ####
    print('Entering Phase B')
    start = time()
    complete_segments: List[Set[Octree]] = []
    for incomplete_segment in tqdm(incomplete_segments):
        # B1a extract boundary voxels
        b_v = extract_boundary_voxels(incomplete_segment)
        # B2 planarity test
        is_planar = check_planarity(incomplete_segment)

        # B3 Refinement (FR or GR)
        refinement(
            is_planar, oc, incomplete_segment, b_v, KDTree)
        complete_segments.append(incomplete_segment.union(b_v))
    if debug:
        colors = [np.random.rand(3) for _ in range(len(complete_segments))]
        draw_complete(complete_segments, points, colors)
    post = time()-start
    save_planes(complete_segments, output_path,
                cloud_path.rsplit('/', 1)[-1].replace('.txt', ''))
    save_time(elapsed, pre, post, output_path, output_path.rsplit(
        '/', 1)[-1], cloud_path.rsplit('/', 1)[-1].replace('.txt', ''))
