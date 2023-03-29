from typing import Union
import argparse
import os
import math as m

import open3d as o3d

def tileo3d(pc: Union[str, o3d.geometry.PointCloud, o3d.t.geometry.PointCloud], n_tiles: int):
    """
    Tiles pointcloud
    If sqrt(n_tiles) is a natural number, all tiles will have equal size, otherwise size will differ
    Assumes z axis is upwards, tiles on x,y

    Parameters
    ----------
    pc
        May be string with path to pointcloud, or o3d.(t) PointCloud object
    n_tiles
        The number of desired tiles
    """


    # check type
    if isinstance(pc, str):
        if not os.path.exists(pc):
            print(f"Can't find file at location {pc}")
            return
        pcd = o3d.io.read_point_cloud(pc)
    elif isinstance(pc, o3d.geometry.PointCloud) or isinstance(pc, o3d.t.geometry.PointCloud):
        pcd = pc

    fl_sqrt = m.isqrt(n_tiles)

    extra_tiles = n_tiles - fl_sqrt**2

    max_bound_xy = pcd.get_max_bound()[:2]
    min_bound_xy = pcd.get_min_bound()[:2]

    


    return

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pointcloud", type=str, required=True)
    parser.add_argument("-n", "--n_tiles", type=int, required=True)
    parser.add_argument("-b", "--buffer", type=int)

    args = parser.parse_args()

    if not os.path.exists(args.pointcloud):
        print("couldnt find pointcloud")
        os._exit(1)
    
    tileo3d(args.pointcloud, args.n_tiles, args.buffer)