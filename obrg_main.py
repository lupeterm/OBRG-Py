import cProfile
from operator import is_
from typing import Dict, List, Set, Tuple
from xml.etree.ElementInclude import include
import open3d as o3d
import numpy as np
from collections import deque
from octree import Octree, dist, get_neighbor_count_same_cluster
from visualization import draw_boundaries, draw_buffer, draw_complete, draw_incomplete, draw_leaf_centers, draw_planar_nplanar, draw_unallocated
from tqdm import tqdm

RES_TH = 0.08  # residual threshold, not explained in paper
D_TH = 0.08
ANG_TH = 0.3  # FIXME
MIN_SEGMENT = 5000  # min points of segment


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
                # for segment in R:
                #     seg_int = segment.intersection(R_c)
                #     if len(seg_int) > 0:
                #         segment.update(R_c)
                #         break
                # else:
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
    # for r in R:
    #     for r2 in R:
    #         if not r.isdisjoint(r2):
    #             r.update(r2)
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
            if d < D_TH:  # 7cm varianz?
                planar += 1
    return (planar / num_points) > 0.8


def fast_refine(O: Octree, R_i: List[Octree], V_b: Set[Octree]) -> None:
    if len(V_b) == 0:
        return
    S = b_v.copy()
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
        # B = get_neighbors(O.leaves, v_j)
        nb_points: Dict[Octree, List] = v_j.get_buffer_zone_points(kdtree)
        # draw_buffer(nb_points, v_j)
        # for index in v_j.indices:
        #     datapoint = O.cloud[index]
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


if __name__ == '__main__':
    # Preparation:
    # read point cloud
    # cloud_path = "whatevs.asc"
    cloud_path = "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/Stanford3dDataset_v1.2_Aligned_Version/TEST/office_4/office_4.txt"
    # cloud_path = "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/Stanford3dDataset_v1.2_Aligned_Version/TEST/WC_1/WC_1.txt"
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
    oc = Octree(points, center=bb.get_center(),
                normals=np.asarray(cloud.normals))
    oc.create(bb.get_max_extent())
    # A1b saliency feature estimation
    print('A1b')

    for leaf in oc.leaves:
        if len(leaf.indices) > 0:
            leaf.calc_n_r()
    # draw_leaf_centers(oc.leaves)
    # A2 voxel based Region Growing
    print('A2')
    cProfile.run('obrg(oc)', sort='tottime')
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
        # draw_boundaries(incomplete_segment, b_v)
        refinement(
            is_planar, oc, incomplete_segment, b_v, KDTree)
        s = set()
        complete_segments.append(incomplete_segment.union(b_v))
        if not is_planar:
            nplanars.append(incomplete_segment)
        else:
            planars.append(incomplete_segment)
    colors = [np.random.rand(3) for _ in range(len(complete_segments))]
    complete_segments.sort(key=lambda x: len(x), reverse=True)
    X = complete_segments[0]

    draw_complete(complete_segments, points, colors)
