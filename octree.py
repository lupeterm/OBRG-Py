from itertools import product
import math
from typing import List
import numpy as np
import open3d as o3d
from scipy import linalg as LA

MAX_LEVEL = 2

centers = []


def dist(point, norm, d):
    d = abs(norm[0]*point[0] + norm[1]*point[1] + norm[2]*point[2] +
            d)/math.sqrt(norm[0]**2 + norm[1]**2 + norm[2]**2)
    return d


class Point:
    def __init__(self, x, y, z) -> None:
        self.x = x
        self.y = y
        self.z = z

    def val(self):
        return self.x, self.y, self.z

    def asarray(self):
        return np.array([self.x, self.y, self.z])

    def round(self):
        return np.around(self.asarray(), decimals=6)

    @staticmethod
    def const(val):
        return Point(val, val, val)


class Octree:
    def __init__(self, cloud=None, center=None) -> None:
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
        self.d = 0
        self.num_nb = 0
        self.is_unallocated = False
        minimum: Point = Point.const(float('inf'))
        maximum: Point = Point.const(float('-inf'))
        for i, point in enumerate(cloud):
            p = Point(*point)
            minimum.x = min(minimum.x, p.x)
            minimum.y = min(minimum.y, p.y)
            minimum.z = min(minimum.z, p.z)
            maximum.x = max(maximum.x, p.x)
            maximum.y = max(maximum.y, p.y)
            maximum.z = max(maximum.z, p.z)
            self.indices.append(i)
            self.points.append(point)
        x = (minimum.x + maximum.x)/2
        y = (minimum.y + maximum.y)/2
        z = (minimum.z + maximum.z)/2
        if center is not None:
            self.center = Point(*center)
        else:
            self.center = Point(x, y, z)
        self.size = max(maximum.x-minimum.x, max(maximum.y -
                        minimum.y, maximum.z - minimum.z))
        self.children = [None] * 8

    def __hash__(self) -> int:
        return hash(tuple(self.center.round()))

    def create(self, minsize: float):
        if len(self.indices) < 3 or self.level > MAX_LEVEL:
            return
        newSize = self.size / 2
        new_centers = [
            Point(self.center.x - newSize, self.center.y -
                  newSize, self.center.z - newSize),
            Point(self.center.x - newSize, self.center.y -
                  newSize, self.center.z + newSize),
            Point(self.center.x - newSize, self.center.y +
                  newSize, self.center.z - newSize),
            Point(self.center.x - newSize, self.center.y +
                  newSize, self.center.z + newSize),
            Point(self.center.x + newSize, self.center.y -
                  newSize, self.center.z - newSize),
            Point(self.center.x + newSize, self.center.y -
                  newSize, self.center.z + newSize),
            Point(self.center.x + newSize, self.center.y +
                  newSize, self.center.z - newSize),
            Point(self.center.x + newSize, self.center.y + newSize, self.center.z + newSize)]
        for c in new_centers:
            centers.append(c)
        self.children = list(map(lambda c: Octree.create_child(
            self, c, newSize, (self.level == MAX_LEVEL)), new_centers))
        for i in self.indices:
            point = Point(*self.cloud[i])
            index = ((point.x > self.center.x) << 2) | (
                (point.y > self.center.y) << 1) | (point.z > self.center.z)
            self.children[index].indices.append(i)
            # self.root.leaves[i] = self.children[index]
        for child in self.children:
            if child.is_leaf and len(child.indices) > 3:
                self.root.leaves.append(child)
            child.create(minsize)

    @staticmethod
    def create_child(parent, center, size, is_leaf=False):
        child = Octree()
        child.cloud = parent.cloud
        child.parent = parent
        child.root = parent.root
        child.level = parent.level+1
        child.center = center
        child.size = size
        child.is_leaf = is_leaf
        child.indices = []
        child.num_nb = 0
        child.is_unallocated = False
        return child

    def calc_n_r(self):
        inliers = np.array([self.cloud[i] for i in self.indices])
        mean = np.mean(inliers, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(inliers)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        self.normal = np.mean(np.asarray(pcd.normals), axis=0)
        p_d = self.center.x*self.normal[0] + self.center.y * \
            self.normal[1] + self.center.z*self.normal[2]
        self.d = p_d
        D = 0
        for inlier in inliers:
            D += (dist(inlier, self.normal, p_d)**2)
        D = D/len(inliers)
        self.residual = D**0.5


leaves: List[Octree] = []


def get_neighbors(leaves: List[Octree], leaf: Octree) -> List[Octree]:
    assert leaf.is_leaf 
    step = leaf.size*2
    neighbor_centers = np.array(list(product([-step, 0, step], repeat=3)))
    neighbor_centers = np.delete(neighbor_centers, 13, axis=0)  # repeat 0,0,0
    # neighbor_centers = np.array(
    #     [[-step, 0, 0], [step, 0, 0], [0, step, 0], [0, -step, 0], [0, 0, step], [0, 0, -step]])
    neighbor_centers += leaf.center.asarray()
    np.around(neighbor_centers, decimals=6, out=neighbor_centers)
    neighbor_centers = set([tuple(x) for x in neighbor_centers])
    nbs = []
    for root_leaf in leaves:
        if tuple(root_leaf.center.round()) in neighbor_centers:
            nbs.append(root_leaf)
            neighbor_centers.remove(tuple(root_leaf.center.round()))
    leaf.num_nb = len(nbs)
    return nbs


if __name__ == '__main__':
    points = np.loadtxt('WC_1.txt', dtype=float, usecols=(0, 1, 2)).tolist()
    bb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(points))
    oc = Octree(points, center=bb.get_center())
    oc.create(bb.get_max_extent())
    # oc.create(5.0)
    leaf_centers = []
    for leaf in leaves:
        leaf_centers.append([leaf.center.x, leaf.center.y, leaf.center.z])

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
