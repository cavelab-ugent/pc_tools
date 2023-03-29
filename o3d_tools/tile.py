from typing import Union
import argparse
import os
import math as m
import numpy as np
import matplotlib.pyplot as plt

import open3d as o3d

def tileo3d(pc: Union[str, o3d.geometry.PointCloud, o3d.t.geometry.PointCloud], n_tiles: int, buffer: int = None, visualization: bool = False):
    """
    Tiles pointcloud (on x,y) into n_tiles
    If sqrt(n_tiles) is a natural number, all tiles will have equal size

    Parameters
    ----------
    pc
        May be string with path to pointcloud, or o3d.(t) PointCloud object
    n_tiles
        The number of desired tiles
    (optional) buffer
        Buffer in meters to add to each tile border
    (optional) visualization
        Visualization of tiles
    """

    # TODO: include buffer
    # TODO: detect longest side and use this one as current y

    # check type
    if isinstance(pc, str):
        if not os.path.exists(pc):
            print(f"Can't find file at location {pc}")
            return
        pcd = o3d.io.read_point_cloud(pc)
    elif isinstance(pc, o3d.geometry.PointCloud) or isinstance(pc, o3d.t.geometry.PointCloud):
        pcd = pc

    if n_tiles <= 1:
        print("n_tiles <= 1")
        return
    
    points = np.asarray(pcd.points)

    fl_sqrt = m.isqrt(n_tiles) # = floor(sqrt(n_tiles))
    # extra tiles to generate
    extra_tiles = n_tiles - fl_sqrt**2

    # get bounds
    max_x, max_y = pcd.get_max_bound()[:2]
    min_x, min_y = pcd.get_min_bound()[:2]

    # check if we need to give both axes an extra division
    if (extra_tiles <= fl_sqrt):
        y_div = fl_sqrt
    else:
        y_div = fl_sqrt+1
        extra_tiles -= fl_sqrt

    y_width = (max_y - min_y) / y_div
    tiles = {}
    tile_bounds = {}
    tile_key = 1

    if buffer is None:
        buffer = 0

    for i in range(y_div):
        # get first slice
        y_slice = points[(points[:,1] <= (min_y + (i+1)*y_width+buffer)) & (points[:,1] >= (min_y + i*y_width-buffer))]

        # divide x into fl_sqrt or fl_sqrt+1 tiles depending on how many extra tiles needed
        x_range = fl_sqrt
        if i < extra_tiles:
            x_range += 1
        x_width = (max_x - min_x)/x_range

        for j in range(x_range):
            # divide into tiles for current y slice
            xy_slice = y_slice[(y_slice[:,0] <= (min_x + (j+1)*x_width+buffer)) & (y_slice[:,0] >= (min_x + j*x_width-buffer))]
            tiles[tile_key] = xy_slice
            tile_bounds[tile_key] = [(min_x + j*x_width, min_x + (j+1)*x_width), (min_y + i*y_width, min_y + (i+1)*y_width)]
            tile_key += 1

    # write output
    out_dir = "tiled/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for tile in tiles:
        o3d.io.write_point_cloud(out_dir+"Tile"+str(tile)+".ply", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tiles[tile])))

    if visualization:
        visualize_bounds([min_x, min_y], [max_x, max_y], tile_bounds)
    return

def visualize_bounds(min_bounds, max_bounds, bound_dict):
    plt.figure()
    plt.axis('equal')

    # draw initial borders first
    plt.hlines(y=min_bounds[1], xmin = min_bounds[0], xmax = max_bounds[0])
    plt.vlines(x=min_bounds[0], ymin = min_bounds[1], ymax = max_bounds[1])

    # draw bounds
    for tile in bound_dict:
        bounds = bound_dict[tile]
        plt.vlines(x = bounds[0][0], ymin = bounds[1][0], ymax = bounds[1][1])
        plt.vlines(x = bounds[0][1], ymin = bounds[1][0], ymax = bounds[1][1])
        plt.hlines(y = bounds[1][0], xmin = bounds[0][0], xmax = bounds[0][1])
        plt.hlines(y = bounds[1][1], xmin = bounds[0][0], xmax = bounds[0][1])
        plt.annotate("Tile "+ str(tile), xy=((bounds[0][0] + bounds[0][1])/2, (bounds[1][0] + bounds[1][1])/2))

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pointcloud", type=str, required=True)
    parser.add_argument("-n", "--n_tiles", type=int, required=True)
    parser.add_argument("-b", "--buffer", type=int)

    args = parser.parse_args()

    if not os.path.exists(args.pointcloud):
        print("couldnt find pointcloud")
        os._exit(1)
    
    tileo3d(args.pointcloud, args.n_tiles, args.buffer, visualization=True)