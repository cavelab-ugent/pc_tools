import open3d as o3d
import time
import argparse
import os
from plyfile import PlyData, PlyElement

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pointcloud", type=str, required=True)

args = parser.parse_args()

if not os.path.exists(args.pointcloud):
    print("couldnt find pointcloud")
    os._exit(1)


t = time.process_time()
pcd = o3d.io.read_point_cloud(args.pointcloud)
t2 = time.process_time()
print(f"03d read took {t2 - t } seconds")


t = time.process_time()
plydata = PlyData.read(args.pointcloud)
t2 = time.process_time()
print(f"plydata read took {t2 - t } seconds")



