
from typing import List, Set
from octree import Octree
import open3d as o3d
import numpy as np


def draw_incomplete(incomplete_segments: List[Set[Octree]],colors):
    clouds = []
    for i, segment in enumerate(incomplete_segments):
        points = []
        pcd = o3d.geometry.PointCloud()
        for leaf in segment:
            for p in leaf.indices:
                points.append(leaf.cloud[p])
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(colors[i])
        clouds.append(pcd)
    o3d.visualization.draw_geometries(clouds)

def draw_complete(complete_segments:List[Set[Octree]], points,colors):
    clouds = []
    for i, segments in enumerate(complete_segments):
        pts = []
        pcd = o3d.geometry.PointCloud()
        for segment in segments:
            for index in segment.indices:
                pts.append(points[index])
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color(colors[i])
        # o3d.visualization.draw_geometries([pcd])
        clouds.append(pcd)
    o3d.visualization.draw_geometries(clouds)

def draw_unallocated(leaves: List[Octree]):
    clouds = []
    for leaf in leaves:
        pts = []
        pcd = o3d.geometry.PointCloud()
        for index in leaf.indices:
            pts.append(leaf.cloud[index])
        pcd.points = o3d.utility.Vector3dVector(pts)
        if not leaf.is_unallocated:
            pcd.paint_uniform_color([0.7,0.7,0.7])
        else:
            pcd.paint_uniform_color([1,0,0])

        clouds.append(pcd)
    o3d.visualization.draw_geometries(clouds)

def draw_boundaries(cluster, boundaries):
    clouds = []
    for leaf in cluster:
        pts = []
        pcd = o3d.geometry.PointCloud()
        for index in leaf.indices:
            pts.append(leaf.cloud[index])
        pcd.points = o3d.utility.Vector3dVector(pts)
        if not leaf in boundaries:
            pcd.paint_uniform_color([0.7,0.7,0.7])
        else:
            pcd.paint_uniform_color([0,0,1])

        clouds.append(pcd)
    o3d.visualization.draw_geometries(clouds)

def draw_planar_nplanar(planar, nplanar):
    clouds = []
    for leaf in nplanar:
        pts = []
        pcd = o3d.geometry.PointCloud()
        for segment in leaf:
            for index in segment.indices:
                pts.append(segment.cloud[index])
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color([1,0,0])
        clouds.append(pcd)
    for leaf in planar:
        pts = []
        pcd = o3d.geometry.PointCloud()
        for segment in leaf:
            for index in segment.indices:
                pts.append(segment.cloud[index])
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color([0,1,0])
        clouds.append(pcd)
    o3d.visualization.draw_geometries(clouds)
