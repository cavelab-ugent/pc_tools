import os
import argparse
import sys
import csv
import time

import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt

def read_csv(file):
    pos = []
    with open(file) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            pos.append([float(i) for i in row[1:]])
    return pos

def boundingrectangleo3d(points, visualisation: bool = False, out_dir = None):
    """
    NOTE: this is slower and not exact due to 3D points being projected onto 2D. For 2D bounding rectangle, use custom boundingrectangle() instead
    Calculates smallest bounding rectangle around xy projections of 3D points using open3d
    Returns 03d.t.geometry.OrientedBoundingBox, and np array of 4 bounding points projected onto xy

    Parameters
    ----------
    points
        array of 3D points
    (optional) visualisation
        whether to show visualisation of projected bounding rectangle
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

    if (visualisation or out_dir):
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
        # create edges:
        for i in range(len(edge_points)-1):
            plt.plot([edge_points[i][0], edge_points[i+1][0]], [edge_points[i][1], edge_points[i+1][1]], 'r-')
        # final line
        plt.plot([edge_points[-1][0], edge_points[0][0]], [edge_points[-1][1], edge_points[0][1]], 'r-')
        if not os.path.exists(out_dir):
            print("Cant find location {out_dir} to save Plotbounds")
        else:
            plt.savefig(os.path.join(out_dir, "PlotBoundso3d.png"))
        if (visualisation):
            plt.show()

    return bbox, projected_corners

def boundingrectangle(points, buffer: int = None, visualisation: bool = False, out_dir = None):
    """
    Calculates smallest bounding rectangle aroung set of 2D points

    Parameters
    ----------
    points
        array of 2D points
    (optional) buffer
        optional buffer to add to edge of bounding rectangle
    (optional) visualisation
        wether to show figure with bounding box
    (optional) out_dir
        location to save figure if provided
    """

    points = np.asarray(points)
    if buffer is None:
        buffer = 0
    
    hull = ConvexHull(points) # scipy fast convex hull

    min_area = np.inf
    final_corners = None

    # TODO: could do this all in matrix multiplications instead of with for loop
    for simplex in hull.simplices:
        start = points[simplex[0]]
        end = points[simplex[1]]

        # get unit direction and normal of current simplex
        direction = np.sum((end, -start), axis=0)
        norm = np.sqrt(np.sum(direction**2))
        unit_dir = direction/norm
        normal_dir = np.asarray([-unit_dir[1], unit_dir[0]])

        # tranform vertices to align current simplex with x axis
        vertices = points[hull.vertices]
        trans_matr = np.vstack((unit_dir, normal_dir)).T
        transformed_vertices = np.matmul(vertices, trans_matr)

        # get minimal and maximal x and y values
        min_vals = np.amin(transformed_vertices, axis=0)
        max_vals = np.amax(transformed_vertices, axis=0)

        max_vals += buffer/2
        min_vals -= buffer/2
        # calculate area
        area = np.prod(max_vals-min_vals)
        # get corner points
        corner_points = np.array([[min_vals[0], min_vals[1]], [min_vals[0], max_vals[1]], [max_vals[0], max_vals[1]], [max_vals[0], min_vals[1]]])

        if area < min_area:
            min_area = area
            # retransform corner points to get edge points using inverse of previous matrix multiplication
            # unit vectors so no renorm, and can just use T
            inv_matrix = trans_matr.T
            final_corners = np.matmul(corner_points, inv_matrix)
    
    if (visualisation or out_dir):
        plt.figure()
        plt.axis('equal')

        for point in points:
            plt.plot(point[0], point[1], 'bo')
        for point in final_corners:
            plt.plot(point[0], point[1], 'ro')
        # create edges:
        for i in range(len(final_corners)-1):
            plt.plot([final_corners[i][0], final_corners[i+1][0]], [final_corners[i][1], final_corners[i+1][1]], 'r-')
        # final line
        plt.plot([final_corners[-1][0], final_corners[0][0]], [final_corners[-1][1], final_corners[0][1]], 'r-')
        if(out_dir):
            if not os.path.exists(out_dir):
                print("Cant find location {out_dir} to save Plotbounds")
            else:
                plt.savefig(os.path.join(out_dir,"PlotBounds.png"))
        if (visualisation):
            plt.show()

    return min_area, final_corners

def calc_rectangle_area(points):
    if len(points) != 4:
        print("more then 4 points")
        return None
    base = points[0]
    dists = []
    # calculate distance with all points, and multiply 2 shortest ones
    # probs not the fastest way but it works
    for point in points[1:,:]:
        dist = np.linalg.norm(point-base, 2)
        dists.append(dist)
    l, w = sorted(dists)[:2]
    return l*w


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # read csv
        csv_file = sys.argv[1]
        pnts = np.asarray(read_csv(csv_file))
        t = time.process_time()
        area, points = boundingrectangle(pnts[:, :2], visualisation= True)
        t2 = time.process_time()
        print(f"Custom bounding rectangle took {t2 - t:.4f} seconds and got area {area:.2f} m2")

        t = time.process_time()
        bbox, points = boundingrectangleo3d(pnts)
        t2 = time.process_time()
        area2 = calc_rectangle_area(points[:, :2])
        print(f"o3d rectangle took {t2 - t:.4f} seconds and got area {area2:.2f} m2")
    else:
        rng = np.random.default_rng()
        points = rng.random((30, 2))
        boundingrectangle(points)


