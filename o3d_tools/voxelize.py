import os
import argparse
import open3d as o3d


def voxelize():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pointcloud", type=str, required=True)
    parser.add_argument("-r", "--resolution", type=float, required=True)

    args = parser.parse_args()

    if not (os.path.exists(args.pointcloud)):
        print("Error: pointcloud not found at given path")
        os._exit(1)

    
    print(f"Reading pointcloud {os.path.basename(args.pointcloud)}")
    pc = o3d.io.read_point_cloud(args.pointcloud)

    print("Voxelizing pointcloud")
    voxelgrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, voxel_size=args.resolution)

    print(f"Writing to {args.pointcloud[:-4] + '_voxelized.ply'}")
    o3d.io.write_voxel_grid(args.pointcloud[:-4] + "_voxelized.ply", voxelgrid)

    return

if __name__ == "__main__":
    voxelize()





