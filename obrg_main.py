from typing import List, Set, Tuple
import open3d as o3d
import numpy as np
from collections import deque
from skspatial.objects import Plane

from octree import Octree, dist, get_neighbors

RES_TH = 20.0  # residual threshold, not explained in paper
ANG_TH = 0.03  # FIXME
MIN_SEGMENT = 200  # min points of segment

def get_cloud(path):
    points = np.loadtxt(cloud_path, dtype=float, usecols=(0, 1, 2)).tolist()
    return points


def unit_vector(n):
    return n/np.linalg.norm(n)


def ang_div(n1, n2):
    v1_u = unit_vector(n1)
    v2_u = unit_vector(n2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def obrg(O: Octree):
    R = []

    A = O.leaves.copy()
    A = deque(sorted(A, key=lambda l: l.residual))
    while len(A) > 0:
        R_c: List[Octree] = []
        S_c: List[Octree] = []
        # "return and remove the voxel having the smallest residual value among set A"
        v_min = A.popleft()
        if v_min.residual > RES_TH:
            break
        R_c.append(v_min)
        S_c.append(v_min)
        for v_i in S_c:
            B_c:List[Octree] = get_neighbors(O.leaves, v_i)
            for v_j in B_c:
                ang_diff = ang_div(v_i.normal, v_j.normal)
                if v_j in A and ang_diff <= ANG_TH:
                    # print("adding to R_c")
                    R_c.append(v_j)
                    A.remove(v_j)
                    if v_j.residual < RES_TH:
                        S_c.append(v_j)
            m = sum([len(l.indices) for l in R_c])
            if m > MIN_SEGMENT:
                R.append(R_c)
            else:
                for l in R_c:
                    l.is_unallocated = True
                R_c.clear()

    return list(reversed(sorted(R, key=lambda x: len(x))))


def extract_boundary_voxels(cluster: List[Octree]):
    return [leaf for leaf in cluster if leaf.num_nb < 8]

def check_planarity(r_i: List[Octree]):
    avg_norm = sum([l.normal for l in r_i]) / len(r_i)
    avg_d  = sum([l.d for l in r_i]) / len(r_i)
    num_points = sum([len(l.indices) for l in r_i])
    planar = 0
    for leaf in r_i:
        for index in leaf.indices:
            d = dist(leaf.cloud[index], avg_norm, avg_d)
            if d < 3:
                planar += 1
    return (planar / num_points) > 0.7

def fast_refine(O:Octree, R_i:List[Octree], V_b:List[Octree]):
    segment = []
    S = V_b.copy()
    if S == []:
        return R_i
    while len(S) > 0:
        v_j = S.pop()
        B = get_neighbors(O.leaves, v_j)
        for v_k in B:
            if v_k.is_unallocated:
                for p_l in v_k.indices:
                    if dist(v_k.cloud[p_l], v_k.normal, v_k.d) < RES_TH:
                        # R_i.append(p_l)
                        v_j.indices.append(p_l)                        
                        S.append(v_k)
        segment.append(v_j)
    return segment

def general_refinement(R_i:List[Octree], b_v:List[Octree]):
    v_b = filter(lambda v: v.is_unallocated, b_v)
    planemaybe = [] # list of points
    for leaf in R_i:
        for inlier in leaf.indices:
            pass # TODO            


if __name__ == '__main__':
    # Preparation:
    # read point cloud
    cloud_path = "WC_1.txt"
    points = get_cloud(cloud_path)
    bb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(points))
    
    # PHASE A

    # A1a voxelization
    oc = Octree(points, center=bb.get_center())
    oc.create(bb.get_max_extent())
    # A1b salience feature estimation 

    for leaf in oc.leaves:
        if len(leaf.indices) > 0:
            leaf.calc_n_r()
        
    # A2 voxel based Region Growing
    incomplete_segments = obrg(oc)
    
    # PHASE B

    # B1a extract boundary voxels
    fast_segments= []
    for incomplete_segment in incomplete_segments:
        b_v = extract_boundary_voxels(incomplete_segment)

        # B2 planarity test
        is_planar = check_planarity(incomplete_segment)

        if is_planar:
            # fast refinement
            fast_segments.append(fast_refine(oc,incomplete_segment, b_v))
        else:
            # general refinement for non planar segments 
            pass #TODO

    clouds = []
    for incomplete_segment in incomplete_segments:
        pts = []
        for leaf in incomplete_segment:
            for inlier in leaf.indices:
                pts.append(points[inlier])
        pcd = o3d.geometry.PointCloud()
        pcd.points= o3d.utility.Vector3dVector(pts)
        # clouds.append(pcd)
        o3d.visualization.draw_geometries([pcd])