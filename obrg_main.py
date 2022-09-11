import cProfile
from typing import List, Set, Tuple
import open3d as o3d
import numpy as np
from octree import Octree
import math
from itertools import product
from collections import deque
from skspatial.objects import Plane
nb_indices = np.array(list(product([-1, 0, 1], repeat=3)))
nb_indices = np.delete(nb_indices, 13, axis=0)  # repeat 0,0,0

RES_TH = 10.0  # residual threshold, not explained in paper
ANG_TH = 1.0  # FIXME
MIN_SEGMENT = 100  # min points of segment


class Leaf:
    def __init__(self, normal, residual, leaf, origin, size, points, grid_index) -> None:
        self.normal = normal
        self.residual = residual
        self.leaf = leaf
        self.origin = origin
        self.size = size
        self.N = len(points)
        self.points = points
        self.grid_index = grid_index

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Leaf):
            return self.origin[0] == __o.origin[0] and self.origin[1] == __o.origin[1] and self.origin[2] == __o.origin[2]
        else:
            return False


leafs: List[Leaf] = []
voxel_to_leaf = dict()

def A1a(pointcloud):
    """create octree from given point cloud"""
    # octree = Octree(pointcloud)
    octree = o3d.geometry.Octree()
    octree.convert_from_point_cloud(pointcloud, size_expand=0.01)
    return octree


def f_traverse(node, node_info):
    early_stop = False
    global pcd, ns, pts, voxel_grid

    if isinstance(node, o3d.geometry.OctreeInternalNode):
        pass
    elif isinstance(node, o3d.geometry.OctreeLeafNode):

        D = 0
        center = np.asarray(node_info.origin)
        normals = []
        for index in node.indices:
            n = ns[index]
            normals.append(n)
        norm_avg = sum(normals) / len(normals)
        p_d = center[0]*norm_avg[0] + center[1]*norm_avg[1] + center[2]*norm_avg[2]

        for index in node.indices:
            p = pts[index]
            d = abs(norm_avg[0]*p[0] + norm_avg[1]*p[1] + norm_avg[2]*p[2] + p_d)/math.sqrt(norm_avg[0]**2 + norm_avg[1]**2 + norm_avg[2]**2)
            D += (d**2)
        D = D/len(normals)
        residual = D**0.5
        grid_coord = voxel_grid.get_voxel(center)
        leaf = Leaf(norm_avg, residual, node, node_info.origin,
                    node_info.size, node.indices, grid_coord)
        voxel_to_leaf[tuple(grid_coord)] = leaf
        leafs.append(leaf)
    else:
        raise NotImplementedError('Node type not recognized!')

    # early stopping: if True, traversal of children of the current node will be skipped
    return early_stop


def get_cloud(path):
    points = np.loadtxt(cloud_path, dtype=float, usecols=(0, 1, 2))
    points = o3d.utility.Vector3dVector(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = points
    return pcd


def get_nb_for_leaf(leaf: Leaf) -> List[Leaf]:
    global voxel_grid, voxels, octree
    # center = leaf.origin
    # center_voxel = voxel_grid.get_voxel(center)
    center_voxel = leaf.grid_index
    nbs = []
    for i in nb_indices:
        nb_i = tuple(i + center_voxel)
        if nb_i in voxels:
            # if nb_i in voxel_to_leaf.keys():
            nbs.append(voxel_to_leaf[nb_i])
            # for leaf in leafs:
            #     if leaf.grid_index == nb_i:
                # # if np.array_equal(leaf.origin, coord):
                # if leaf.origin[0] == coord[0] and leaf.origin[1] == coord[1] and leaf.origin[2] == coord[2]:
            #         nbs.append(leaf)
            #         break
            # else:
            #     # print('doesnt match')
            #     pass
    return nbs


def get_neighbors(voxels, v_index):
    nbs = []
    for i in nb_indices:
        nb_i = i + v_index
        if tuple(nb_i) in voxels:
            nbs.append(tuple(nb_i))
    return nbs


def unit_vector(n):
    return n/np.linalg.norm(n)


def ang_div(n1, n2):
    v1_u = unit_vector(n1)
    v2_u = unit_vector(n2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


UNALLOC = []


def obrg():
    global residuals, pcd
    R = []

    A = leafs.copy()
    A = deque(sorted(A, key=lambda l: l.residual))
    while len(A) > 0:
        R_c: List[Leaf] = []
        S_c: List[Leaf] = []
        # "return and remove the voxel having the smallest residual value among set A"
        v_min = A.popleft()
        if v_min.residual > RES_TH:
            break
        R_c.append(v_min)
        S_c.append(v_min)
        for v_i in S_c:
            current_seed = v_i
            # cProfile.run('get_nb_for_leaf(current_seed)', sort='tottime')
            B_c = get_nb_for_leaf(current_seed)
            for v_j in B_c:
                ang_diff = ang_div(v_i.normal, v_j.normal)
                if v_j in A and ang_diff <= ANG_TH:
                    # print("adding to R_c")
                    R_c.append(v_j)
                    A.remove(v_j)
                    if v_j.residual < RES_TH:
                        S_c.append(v_j)
            m = sum([l.N for l in R_c])
            if m > MIN_SEGMENT:
                R.append(R_c)
            else:
                for l in R_c:
                    UNALLOC.append(l)

    return list(reversed(sorted(R, key=lambda x: len(x))))


def extact_boundary_voxels():
    return []


def fast_refine(R_i, V_b):
    S = []  # = V_b
    while len(S) > 0:
        v_j = S.pop()
        B = get_nb_for_leaf(v_j)
        for v_k in B:
            if v_k in UNALLOC:
                for p_l in v_k.points:  # TODO add points to datastructure
                    if dist(residuals[p_l], PLANE) < D_TH:
                        # R_i.append(p_l)
                        # S.append(v_k)
                        pass


if __name__ == '__main__':
    cloud_path = "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/Stanford3dDataset_v1.2_Aligned_Version/TEST/WC_1/WC_1.txt"
    # cloud_path = "testcloud.txt"
    # Preparation:
    # read point cloud
    pcd = get_cloud(cloud_path)
    residuals = [0.0] * len(pcd.points)
    N = [0.0] * len(pcd.points)
    # calculate normals and residual values for each voxel
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    bb = pcd.get_axis_aligned_bounding_box()
    bbx = bb.get_extent()
    ns = np.asarray(pcd.normals)
    pts = np.asarray(pcd.points)

    #### PHASE A ####
    # A1a.   Voxelization of point cloud
    octree = o3d.geometry.Octree(max_depth=2)
    octree.convert_from_point_cloud(pcd, size_expand=0.0)
    voxel_grid = octree.to_voxel_grid()
    # R = dict()
    # for point in pcd.points:
    #     v = tuple(voxel_grid.get_voxel(point))

    # A1b.   Saliency feature estimation for voxels
    octree.traverse(f_traverse)
    points = np.array([l.origin for l in leafs])
    nrs = [leaf.normal for leaf in leafs]
    points = o3d.utility.Vector3dVector(points)
    vgpcd = o3d.geometry.PointCloud()
    vgpcd.points = points
    vgpcd.normals = o3d.utility.Vector3dVector(nrs)
    # o3d.visualization.draw_geometries([vgpcd])
    min_size = min(leafs, key=lambda l: l.size).size
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        vgpcd, voxel_size=min_size)
    voxels = set([tuple(v.grid_index) for v in voxel_grid.get_voxels()])

    # o3d.visualization.draw_geometries([voxel_grid])

    residuals = np.array(residuals)
    # A2.    Voxel-based RG
    # => Incomplete clusters

    incomplete_clusters = obrg()

    #### PHASE B ####
    # B1a.   Extract boundary voxels NOTE wat
    # for r_i in incomplete_clusters:
    p = []
    for leaf in incomplete_clusters[0]:
        for pt in leaf.points:
            p.append(pts[pt])

    PC = o3d.geometry.PointCloud()
    PC.points = o3d.utility.Vector3dVector(p)
    PC.colors = o3d.utility.Vector3dVector(np.full_like(p, [0,0,1]))


    p2 = []
    for leaf in incomplete_clusters[1]:
        for pt in leaf.points:
            p2.append(pts[pt])

    PC2 = o3d.geometry.PointCloud()
    PC2.points = o3d.utility.Vector3dVector(p2)
    PC2.colors = o3d.utility.Vector3dVector(np.full_like(p2, [1,0,0]))

    pcd.colors = o3d.utility.Vector3dVector(np.full_like(pts, [0,1,0]))
    

    o3d.visualization.draw_geometries([pcd, PC, PC2])

    # B1b.   find points in clusters vicinity
    # B2.    check for planarity
    # B3.    Refinement (fast or general)

    # => complete clusters

    # Write clusters to file
