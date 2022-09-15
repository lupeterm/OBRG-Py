from inspect import isdatadescriptor
from itertools import product
import math
from typing import List
import numpy as np
import open3d as o3d
from scipy import linalg as LA

MAX_LEVEL = 7
centers = []

corner_indices = np.array([v for v in list(product([-1, 0, 1], repeat=3)) if 0 not in v])
NEIGHBOR_INDICES = np.array(list(product([-1, 0, 1], repeat=3)))
NEIGHBOR_INDICES = np.delete(NEIGHBOR_INDICES, 13, axis=0)


def dist(point, norm, d):
    d = abs(norm[0]*point[0] + norm[1]*point[1] + norm[2]*point[2] +
            -d)/math.sqrt(norm[0]**2 + norm[1]**2 + norm[2]**2)
    return d


class Octree:
    def __init__(self, cloud=None, center=None, normals=None) -> None:
        if cloud is None:
            return
        self.cloud = cloud
        self.root = self
        self.parent = None
        self.level = 0
        self.leaves: List[Octree] = []
        self.indices = []
        self.points = []
        self.is_leaf = False
        self.normal = self.residual = 0
        self.leaf_centers = dict()
        if normals is not None:
            self.normals = normals
        self.d = 0
        self.num_nb = 0
        self.is_unallocated = False
        self.residuals = []
        minimum = [float('inf')]*3
        maximum = [-float('inf')]*3
        for i, point in enumerate(cloud):
            x,y,z = point
            minimum[0] = min(minimum[0], x)
            minimum[1] = min(minimum[1], y)
            minimum[2] = min(minimum[2], z)
            maximum[0] = max(maximum[0], x)
            maximum[1] = max(maximum[1], y)
            maximum[2] = max(maximum[2], z)
            self.indices.append(i)
            self.points.append(point)
        x = (minimum[0] + maximum[0])/2
        y = (minimum[1] + maximum[1])/2
        z = (minimum[2] + maximum[2])/2
        if center is not None:
            self.center = np.array([*center])
        else:
            self.center = np.array([x, y, z])
        self.size = max(maximum[0]-minimum[0], max(maximum[1] -
                        minimum[1], maximum[2] - minimum[2]))
        self.children = [None] * 8

    def __hash__(self) -> int:
        return hash(tuple(np.around(np.array(self.center),decimals=4)))

    def __eq__(self, __o: object) -> bool:
        return id(self) == id(__o)

    def create(self, minsize: float):
        if len(self.indices) < 3 or self.level > MAX_LEVEL:
            return
        newSize = self.size / 2
        new_centers = [
            [self.center[0] - newSize, self.center[1] -
                  newSize, self.center[2] - newSize],
            [self.center[0] - newSize, self.center[1] -
                  newSize, self.center[2] + newSize],
            [self.center[0] - newSize, self.center[1] +
                  newSize, self.center[2] - newSize],
            [self.center[0] - newSize, self.center[1] +
                  newSize, self.center[2] + newSize],
            [self.center[0] + newSize, self.center[1] -
                  newSize, self.center[2] - newSize],
            [self.center[0] + newSize, self.center[1] -
                  newSize, self.center[2] + newSize],
            [self.center[0] + newSize, self.center[1] +
                  newSize, self.center[2] - newSize],
            [self.center[0] + newSize, self.center[1] + newSize, self.center[2] + newSize]]
        for c in new_centers:
            centers.append(c)
        self.children = list(map(lambda c: Octree.create_child(
            self, c, newSize, (self.level == MAX_LEVEL)), new_centers))
        for i in self.indices:
            point = np.array(self.cloud[i])
            index = ((point[0]> self.center[0]) << 2) | (
                (point[1] > self.center[1]) << 1) | (point[2] > self.center[2])
            self.children[index].indices.append(i)
            # self.root.leaves[i] = self.children[index]
        for child in self.children:
            if child.is_leaf and len(child.indices) > 3:
                self.root.leaves.append(child)
                self.root.leaf_centers[tuple(np.around(child.center,decimals=4))] = child
            child.create(minsize)

    @staticmethod
    def create_child(parent, center, size, is_leaf=False):
        global IDS
        child = Octree()
        child.cloud = parent.cloud
        child.parent = parent
        child.root = parent.root
        child.level = parent.level+1
        child.center = center
        child.size = size
        child.is_leaf = is_leaf
        child.indices = []
        child.leaf_centers = parent.root.leaf_centers
        if is_leaf:
            child.residuals = []
        child.num_nb = 0
        child.is_unallocated = False
        child.normals = parent.normals
        return child

    def calc_n_r(self):
        inliers = np.array([self.cloud[i] for i in self.indices])
        normals = np.array(list(map(lambda i: self.normals[i],self.indices)))
        self.normal = np.mean(normals, axis=0)
        p_d = self.center[0]*self.normal[0] + self.center[1] * \
            self.normal[1] + self.center[2]*self.normal[2]
        self.d = p_d
        D = 0
        for inlier in inliers:
            d = dist(inlier, self.normal, p_d)
            self.residuals.append(d)
            D += (d**2)
        # print(np.mean(np.array(self.residuals), axis=0))
        D = D/len(inliers)
        self.residual = D**0.5

    def find_leaf_is_allocated(self, index):
        for leaf in self.root.leaves:
            if index in leaf.indices:
                return leaf.is_unallocated
        return False

    def get_buffer_zone_points(self, kdtree):
        corners = corner_indices * self.size
        corners += self.center
        buffer_points = dict()
        B = self.get_neighbors()
        for nb in B:
            if not nb.is_unallocated:
                continue
            buffer_points[nb] = set()
            for index in nb.indices:
                buffer_points[nb].add(index)
        return buffer_points

    def draw(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            [leaf.cloud[p] for p in leaf.indices])
        o3d.visualization.draw_geometries([pcd])

    def get_neighbors(self):
        global NEIGHBOR_INDICES
        assert self.is_leaf
        step = self.size*2
        neighbors = NEIGHBOR_INDICES * step
        neighbors += self.center
        neighbor_centers = set([tuple(np.around(x,decimals=4)) for x in neighbors])
        nbs = []
        for nb in neighbor_centers:
            if nb in self.leaf_centers.keys():
                    nbs.append(self.leaf_centers[nb])
        # for root_leaf in leaves:
        #     rounded_center = tuple(np.around(root_leaf.center, decimals=4))
        #     if  rounded_center in neighbor_centers:
        #         nbs.append(root_leaf)
        #         neighbor_centers.remove(rounded_center)
        self.num_nb = len(nbs)
        return nbs
leaves: List[Octree] = []

def get_neighbor_count_same_cluster(leaf, cluster_center):
    global NEIGHBOR_INDICES
    assert leaf.is_leaf
    step = leaf.size*2
    neighbors = NEIGHBOR_INDICES * step
    neighbors += leaf.center
    neighbors.round(decimals=6)
    neighbor_centers = set([tuple(np.around(x,decimals=6)) for x in neighbors])
    nbs = set()
    for center in neighbor_centers:
        if center in cluster_center:
            nbs.add(center)
    return len(nbs)




if __name__ == '__main__':
    points = np.loadtxt('WC_1.txt', dtype=float, usecols=(0, 1, 2)).tolist()
    bb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(points))
    oc = Octree(points, center=bb.get_center())
    oc.create(bb.get_max_extent())
    # oc.create(5.0)
    leaf_centers = []
    for leaf in leaves:
        leaf_centers.append([leaf.center[0], leaf.center[1], leaf.center[2]])

    print(leaves[0].calc_n_r())
    exit()
    nb = []
    for pivot in leaves[::5]:
        pcd = o3d.geometry.PointCloud()
        step = pivot.size*2
        neighbor_centers = np.array(
            [[-step, 0, 0], [step, 0, 0], [0, step, 0], [0, -step, 0], [0, 0, step], [0, 0, -step]])
        neighbor_centers += pivot.center.asarray()
        nbs = get_neighbors(leaves, pivot)

        # np.savetxt('vector.txt', np.array(leaf_centers), delimiter=' ')
        pcd.points = o3d.utility.Vector3dVector(nbs)
        nb.append(pcd)
        # pcd2.points = o3d.utility.Vector3dVector(leaf_centers)
        # pcd2.paint_uniform_color([0.7,0.7,0.7])
        # pcd.points = o3d.utility.Vector3dVector(nbs)
    o3d.visualization.draw_geometries(nb)
