import cProfile
from operator import is_
from typing import Dict, List, Set, Tuple
import open3d as o3d
import numpy as np
from collections import deque
from octree import Octree, dist, get_neighbor_count_same_cluster, get_neighbors
from visualization import draw_boundaries, draw_complete, draw_incomplete, draw_planar_nplanar, draw_unallocated
from tqdm import tqdm

RES_TH = 0.1  # residual threshold, not explained in paper
D_TH = 0.05
ANG_TH = 0.3  # FIXME
MIN_SEGMENT = 1000  # min points of segment


def get_cloud(path):
    points = np.loadtxt(cloud_path, dtype=float, usecols=(0, 1, 2)).tolist()
    return points


def unit_vector(n):
    return n/np.linalg.norm(n)


def ang_div(n1, n2):
    v1_u = unit_vector(n1)
    v2_u = unit_vector(n2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def obrg(O: Octree) -> List[Set[Octree]]:
    R: List[Set[Octree]] = list()
    a = O.leaves
    a.sort(key=lambda x: x.residual)
    A = deque(a)
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
            B_c = get_neighbors(O.leaves, v_i)
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
                for segment in R:
                    seg_int = segment.intersection(R_c)
                    if len(seg_int) > 0:
                        segment.update(R_c)
                        break
                else:
                    R.append(R_c)
            else:
                for l in R_c:
                    l.is_unallocated = True
    return sorted(R, key=lambda x: len(x), reverse=True)


def extract_boundary_voxels(cluster: Set[Octree]) -> Set[Octree]:
    cluster_centers = [tuple(np.around(voxel.center,decimals=6)) for voxel in cluster]
    boundaries = set([leaf for leaf in cluster if get_neighbor_count_same_cluster(leaf,cluster_centers)])
    return boundaries        


def check_planarity(r_i: Set[Octree]) -> bool:
    avg_norm = sum([l.normal for l in r_i]) / len(r_i)
    avg_d = sum([l.d for l in r_i]) / len(r_i)
    num_points = sum([len(l.indices) for l in r_i])
    planar = 0
    ds = []
    for leaf in r_i:
        for index in leaf.indices:
            d = dist(leaf.cloud[index], avg_norm, avg_d)
            ds.append(d)
            if d < D_TH:  # 7cm varianz? 
                planar += 1
    return (planar / num_points) > 0.8


def fast_refine(O: Octree, R_i: List[Octree], V_b: Set[Octree]):
    S = b_v.copy()
    norm_R_i = sum([l.normal for l in R_i]) / len(R_i)
    d_R_i = sum([l.d for l in R_i]) / len(R_i)
    to_be_added:Set[int] = set()
    visited = set()
    while len(S) > 0:
        v_j = S.pop()
        visited.add(v_j)
        B = get_neighbors(O.leaves, v_j)
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

def general_refinement(O:Octree, R_i: List[Octree], b_v: List[Octree], kdtree) -> Set[int]:
    S: Set[Octree] = set(b_v)
    visited = set()
    to_add: Set[int] = set()
    while len(S) > 0:
        v_j = S.pop()
        # B = get_neighbors(O.leaves, v_j)
        nb_points = v_j.get_buffer_zone_points(kdtree)
        for nb, buffer_points in nb_points.items():
            for buffer_index in buffer_points:
                datapoint = v_j.cloud[buffer_index]
                distance = dist(datapoint, v_j.normal, v_j.d)
                if distance < D_TH:
                    v_j.indices.append(buffer_index)
                    to_add.add(buffer_index)
                    nb.indices.remove(buffer_index)
    return to_add


def refinement(is_planar, oc, incomplete_segment, b_v, kdtree):
    if is_planar:
        # fast refinement
        print('planar!')
        fast_refine(oc, incomplete_segment, b_v)
    else:
        print('not planar')
        # general refinement for non planar segments
        # segment = general_refinement(oc, incomplete_segment, b_v, kdtree)



if __name__ == '__main__':
    # Preparation:
    # read point cloud
    # cloud_path = "WC_1.txt"
    cloud_path = "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/Stanford3dDataset_v1.2_Aligned_Version/TEST/WC_1/WC_1.txt"
    points = get_cloud(cloud_path)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    bb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(points))
    KDTree = o3d.geometry.KDTreeFlann(cloud)
    # PHASE A
    print('A1a')
    # A1a voxelization
    norms = np.asarray(cloud.normals)
    oc = Octree(points, center=bb.get_center(), normals=np.asarray(cloud.normals))
    oc.create(bb.get_max_extent())
    # A1b saliency feature estimation
    print('A1b')

    for leaf in oc.leaves:
        if len(leaf.indices) > 0:
            leaf.calc_n_r()

    # A2 voxel based Region Growing
    print('A2')
    incomplete_segments = obrg(oc)
    np.random.seed(0)
    colors = [np.random.rand(3) for _ in range(len(incomplete_segments))]
    # colors = [[0,0,0]]* len(incomplete_segments)
    # PHASE B
    print('Ab')

    draw_incomplete(incomplete_segments, colors)
    # draw_unallocated(oc.leaves)

    # B1a extract boundary voxels
    complete_segments: List[Set[Octree]] = []
    planars = []
    nplanars = []
    for incomplete_segment in tqdm(incomplete_segments):
        b_v = extract_boundary_voxels(incomplete_segment)
        # B2 planarity test
        is_planar = check_planarity(incomplete_segment)

        refinement(
            is_planar, oc, incomplete_segment, b_v, KDTree)
        s = set()
        complete_segments.append(incomplete_segment.union(b_v))
        if not is_planar:
            nplanars.append(incomplete_segment)
        else:
            planars.append(incomplete_segment)
        # if is_planar:
        #     new_segment = set()
        #     for l in incomplete_segment:
        #         for p in l.indices:
        #             new_segment.add(p)
        #     for p in to_be_added:
        #         new_segment.add(p)
        #     complete_segments.append(new_segment)
    draw_planar_nplanar(planars, [])
    colors = [np.random.rand(3) for _ in range(len(complete_segments))]
    complete_segments.sort(key=lambda x: len(x), reverse=True)
    X = complete_segments[0]

    draw_complete(complete_segments, points, colors)