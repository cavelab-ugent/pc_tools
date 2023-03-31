import os
import argparse
import sys
import csv

import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt

def read_csv(file):
    pos = []
    with open(file) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            pos.append([float(i) for i in row[1:]])
    return pos

def boundingrectangleo3d(points):
    """
    Calculates smallest bounding rectangle aroung xy projections of 3D points using open3d
    Returns 03d.t.geometry.OrientedBoundingBox, and np array of 4 bounding points projected onto xy

    Parameters
    ----------
    points
        array of 3D points
    """
    points = np.array(points)

    device = o3d.core.Device("CPU:0")
    dtype_f32 = o3d.core.float32
    pcd = o3d.t.geometry.PointCloud(device)

    pcd.point.positions = o3d.core.Tensor(points, dtype_f32, device)

    bbox = pcd.get_oriented_bounding_box()
    
    # open3d rotation is inversed for some reason, so reverse manually
    rot_inv = bbox.rotation.T()
    bbox = bbox.rotate(rot_inv)
    bbox = bbox.rotate(rot_inv)

    edge_points = bbox.get_box_points()
    # project onto xy

    ep_np = edge_points.numpy()

    max_idxs = np.argmax(ep_np, axis=0)
    min_idxs = np.argmin(ep_np, axis=0)

    projected_corners = ep_np[np.concatenate((max_idxs[:2], min_idxs[:2]), axis=0)]

    visualize_bounding_rectangle(points, bbox)

    return bbox, projected_corners

def visualize_bounding_rectangle(points, bbox: o3d.t.geometry.OrientedBoundingBox):
    plt.figure()
    plt.axis('equal')

    edge_points = bbox.get_box_points()
    # project bbox to 
    ep_np = edge_points.numpy()

    max_idxs = np.argmax(ep_np, axis=0)
    min_idxs = np.argmin(ep_np, axis=0)

    edge_points = ep_np[np.concatenate((max_idxs[:2], min_idxs[:2]), axis=0)]

    for point in points:
        plt.plot(point[0], point[1], 'bo')
    for point in edge_points:
        plt.plot(point[0], point[1], 'ro')
    plt.show()

    return


def boundingrectangle(points):
    """
    Calculates smallest bounding rectangle aroung set of 2D points

    Parameters
    ----------
    points
        array of 2D points
    """
    
    # hull = ConvexHull(points) (scipy fast convex hull)

    # algo: one of edges of convex hull is part of edge of bounding rectangle
    # loop over convex hull edges, use dot transform so edge is horizontal, calculate edge aligned bounding box and save area
    # smallest of these rectangles is smallest bounding rectangle



if __name__ == "__main__":
    # read csv

    csv_file = sys.argv[1]
    pnts = np.asarray(read_csv(csv_file))

    boundingrectangleo3d(pnts)

