import os
import argparse
from typing import Union

import numpy as np
import open3d as o3d
import pptk


def visualizepptk(pc: Union[str, o3d.geometry.PointCloud, o3d.t.geometry.PointCloud]):
    """
    Visualizes pointcloud using pptk

    Note: pptk requires python 3.7 or lower, viewer doesn't work in WSL/Ubuntu

    Parameters
    ----------
    pc
        May be string with path to pointcloud, or o3d.(t) PointCloud object
    """

    # check type
    if isinstance(pc, str):
        if not os.path.exists(pc):
            print(f"Can't find file at location {pc}")
            return
        pcd = o3d.io.read_point_cloud(pc)
        points = np.asarray(pcd.points)
    elif isinstance(pc, o3d.geometry.PointCloud) or isinstance(pc, o3d.t.geometry.PointCloud):
        pcd = pc
        points = np.asarray(pcd.points)

    # Visualize point cloud
    v = pptk.viewer(points)
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pointcloud", type=str, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.pointcloud):
        print("couldnt find pointcloud")
        os._exit(1)
    
    visualizepptk(args.pointcloud)