from typing import Union
import argparse
import os
import math as m
import numpy as np
import matplotlib.pyplot as plt

import open3d as o3d

from boundingrectangle import boundingrectangle


def tile_from_corner_points(corners, pc: Union[str, o3d.geometry.PointCloud, o3d.t.geometry.PointCloud], size: int = 10, buffer:int = None, exact_size: bool = False, visualization: bool = False, out_dir: str = None):
    """
    Tiles pointcloud (on x,y) into tiles of given size, given xy coordinates of bounding rectangle
    The bool exact_size indicates wether exact given size is used: 
        if True exact squares will be made and any left over parts will be smaller tiles.
        if False the closest length to divide each side into exact squares will be used.
    Parameters
    ----------
    corners
        2D np array with 4 xy coordinates
    pc
        May be string with path to pointcloud, or o3d.(t) PointCloud object
    size
        Size of one side of tile in metres 
    (optional) buffer
        Buffer in meters to add to each tile border
    (optional) exact_size
        Wether to use exact given tile size
    (optional) visualization
        Visualization of tiles
    (optional) out_dir
        out_dir to save visualization of tiles
    """
    if isinstance(pc, str):
        if not os.path.exists(pc):
            print(f"Can't find file at location {pc}")
            return
        pcd = o3d.io.read_point_cloud(pc)
    elif isinstance(pc, o3d.geometry.PointCloud) or isinstance(pc, o3d.t.geometry.PointCloud):
        pcd = pc
    else:
        print(f"Can't read pointcloud {pc}")
        return
    if not buffer:
        buffer = 0
    # get xy matrix to rotate rectangle
    y_min_crnr = corners[np.argmin(corners, axis=0)[1]]
    x_max_crnr = corners[np.argmax(corners, axis=0)[0]]
    theta = m.atan2((x_max_crnr[1]-y_min_crnr[1] ), (x_max_crnr[0] - y_min_crnr[0]))
    rot_matr_2D = np.array([[m.cos(theta), -m.sin(theta)], [m.sin(theta), m.cos(theta)]])

    # rotated xy bounding box
    rotated_corners = np.matmul(corners, rot_matr_2D)

    # tile rotated bounding box
    tiles = create_tiles_aligned_corners(rotated_corners, size=size, buffer=buffer, exact=exact_size)
    inv_rotation = rot_matr_2D.T

    tiled_pcs = []

    rotated_tiles = []

    for tile in tiles: 
        # rotate tile back
        rotated_tile = np.matmul(tile, inv_rotation)
        rotated_tiles.append(rotated_tile)

        # create bounding box from rotated tile
        LARGE_Z = 1000000

        top_corners = np.hstack((rotated_tile, np.asarray([[LARGE_Z]]*4)))
        bot_corners = np.hstack((rotated_tile, np.asarray([[-LARGE_Z]]*4)))

        if isinstance(pcd, o3d.geometry.PointCloud):
            tile_corners_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack((top_corners, bot_corners))))
        elif isinstance(pcd, o3d.t.geometry.PointCloud):
            tile_corners_pc = o3d.t.geometry.PointCloud(o3d.core.Tensor(np.vstack((top_corners, bot_corners))))
        else:
            return None

        tile_crop_bbox = tile_corners_pc.get_oriented_bounding_box()

        cropped_pc = pcd.crop(tile_crop_bbox)

        tiled_pcs.append(cropped_pc)
    
    if (visualization or out_dir) :
        visualize_rectangles([corners, *tiles, rotated_corners, *rotated_tiles], visualization=visualization, out_dir=out_dir)

    return tiled_pcs

def tile_o3d_xy(pc: Union[str, o3d.geometry.PointCloud, o3d.t.geometry.PointCloud], size: int = 10, buffer: int = None, edge_buffer: int = None, exact_size: bool = False, visualization: bool = False, out_dir: str = None):
    """
    Tiles pointcloud into tiles of given size, by projecting all points onto xy, finding smallest bounding rectangle and tiling on this rectangle

    The bool exact_size indicates wether exact given size is used: 
        if True exact squares will be made and any left over parts will be smaller tiles.
        if False the closest length to divide each side into exact squares will be used.
    Parameters
    ----------
    pc
        May be string with path to pointcloud, or o3d.(t) PointCloud object
    size
        Size of one side of tile in metres 
    (optional) buffer
        Buffer in meters to add to each tile border
    (optional) exact_size
        Wether to use exact given tile size
    (optional) visualization
        Visualization of tiles
    (optional) out_dir
        out_dir to save visualization of tiles
    """

    if isinstance(pc, str):
        if not os.path.exists(pc):
            print(f"Can't find file at location {pc}")
            return
        pcd = o3d.io.read_point_cloud(pc)
        xy_points = np.asarray(pcd.points)[:, :2]
    elif isinstance(pc, o3d.geometry.PointCloud):
        pcd = pc
        xy_points = np.asarray(pcd.points)
    elif isinstance(pc, o3d.t.geometry.PointCloud):
        pcd = pc
        xy_points = pcd.point.positions.numpy()[:, :2]
    else:
        print(f"Can't read pointcloud {pc}")
        return None
    if not buffer:
        buffer = 0

    _, corners = boundingrectangle(xy_points, edge_buffer)

    return tile_from_corner_points(corners, pcd, size=size, buffer=buffer, exact_size=exact_size, visualization=visualization, out_dir=out_dir)

def create_tiles_aligned_corners(corner_points, size, buffer, exact):
    """
    Tiles rectangle with tiles of given size and given overlap
    NOTE: assumes corner_points are aligned with xy
    
    The bool exact_size indicates wether exact given size is used: 
        if True exact squares will be made and any left over parts will be smaller tiles.
        if False the closest length to divide each side into exact squares will be used.

    Returns
    ----------
    tiles:
        list of xy corner points in format [[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]]
    """

    # get min and max x and y
    max = np.amax(corner_points, axis=0)[:2]
    min = np.amin(corner_points, axis=0)[:2]

    lengths = max - min

    tiles = [] # format: each tile: np.array of [[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]]
    
    if exact:
        # slice into tiles of exact given length, then append smaller leftover tiles
        n_tiles = np.ceil((lengths - buffer) / ( size - buffer ))
        # bottom left corners
        x_crnrs = [(min[0] + i*(size-buffer)) for i in range(int(n_tiles[0]))]
        y_crnrs = [(min[1] + i*(size-buffer)) for i in range(int(n_tiles[1]))]
        for i in range(len(x_crnrs)-1):
            for j in range(len(y_crnrs)-1):
                tiles.append([[x_crnrs[i],y_crnrs[j]], [x_crnrs[i],y_crnrs[j+1]],[x_crnrs[i+1],y_crnrs[j]],[x_crnrs[i+1],y_crnrs[j+1]]])
        # append all small boxes left on y side
        for i in range(len(x_crnrs)-1):
            tiles.append([[x_crnrs[i],y_crnrs[-1]], [x_crnrs[i],max[1]],[x_crnrs[i+1],y_crnrs[-1]],[x_crnrs[i+1], max[1]]])
        # append all small boxes left on x side
        for j in range(len(y_crnrs)-1):
            tiles.append([[x_crnrs[-1],y_crnrs[j]], [x_crnrs[-1],y_crnrs[j+1]],[max[0],y_crnrs[j]],[max[0],y_crnrs[j+1]]])
        # append smallest box with xy leftover
        tiles.append([[x_crnrs[-1],y_crnrs[-1]], [x_crnrs[-1],max[1]],[max[0],y_crnrs[-1]],[max[0],max[1]]])
    
    else:
        # use size where we get exact same rectangles for each tile
        # recalculate size of tiles in each direction
        n_tiles = (lengths - buffer) / ( size - buffer )
        # can't divide into 0 tiles
        n_tiles = np.where(n_tiles == 0, 1, n_tiles)
        sizes = (lengths - buffer) / np.round(n_tiles) + buffer
        print(f"Actual sizes used: x: {sizes[0]}, y: {sizes[1]}")
        tiles_xy = np.divide(lengths, sizes)
        # bottom left corners
        x_crnrs = [(min[0] + i*(sizes[0]-buffer)) for i in range(int(tiles_xy[0]))]
        y_crnrs = [(min[1] + i*(sizes[1]-buffer)) for i in range(int(tiles_xy[1]))]
        for i in range(len(x_crnrs)):
            for j in range(len(y_crnrs)):
                tiles.append([[x_crnrs[i],y_crnrs[j]], [x_crnrs[i],y_crnrs[j] + sizes[1]],[x_crnrs[i] + sizes[0],y_crnrs[j]],[x_crnrs[i] + sizes[0],y_crnrs[j] + sizes[1]]])

    return tiles

def visualize_rectangles(rectangles, visualization:bool = False, out_dir: str = None):
    plt.figure()
    plt.axis('equal')
    colors = ['r', 'b', 'g', 'k', 'c', 'm', 'y']
    for j, rectangle in enumerate(rectangles):
        for point in rectangle:
            plt.plot(point[0], point[1], 'ro')
        # create edges: sort based on distance with one of points, two closest points are neighbours
        anchor = rectangle[0]
        sorted_corners = sorted(rectangle[1:], key = lambda x : m.sqrt((x[0]-anchor[0])**2 + (x[1]-anchor[1])**2))
        # connect anchor with 2 closest points
        plt.plot([anchor[0], sorted_corners[0][0]], [anchor[1], sorted_corners[0][1]], '-', color=colors[j % len(colors)])
        plt.plot([anchor[0], sorted_corners[1][0]], [anchor[1], sorted_corners[1][1]], '-', color=colors[j % len(colors)])
        # connect farthest point with 2 other points
        plt.plot([sorted_corners[2][0], sorted_corners[0][0]], [sorted_corners[2][1], sorted_corners[0][1]], '-', color=colors[j % len(colors)])
        plt.plot([sorted_corners[2][0], sorted_corners[1][0]], [sorted_corners[2][1], sorted_corners[1][1]], '-', color=colors[j % len(colors)])

        min_x, min_y = np.amin(rectangle, axis=0)
        max_x, max_y = np.amax(rectangle, axis=0)

        plt.annotate("T" + str(j), xy= ((max_x + min_x)/2, (max_y + min_y)/2))
    if out_dir:
        if not os.path.exists(out_dir):
                print("Cant find location {out_dir} to save Plotbounds")
        else:
            plt.savefig(os.path.join(out_dir,"PlotBounds.png"))
    if visualization:
        plt.show()

def project(bboxs):
    plt.figure()
    plt.axis('equal')
    colors = ['r', 'b', 'g', 'k', 'c', 'm', 'y']
    # draw bounds
    for i, bbox in enumerate(bboxs):
        edges = bbox.get_box_points()
        if isinstance(bbox, o3d.t.geometry.OrientedBoundingBox):
            edges = edges.numpy()
        elif isinstance(bbox, o3d.geometry.OrientedBoundingBox):
            edges = np.asarray(edges)

        # project onto xy
        for edge in edges:
            plt.scatter(edge[0], edge[1], color=colors[i])

    plt.show()

def visualize_bounds(rotated_bboxs):
    plt.figure()
    plt.axis('equal')

    # draw bounds
    for i, bbox in enumerate(rotated_bboxs):
        edges = bbox.get_box_points()
        if isinstance(bbox, o3d.t.geometry.OrientedBoundingBox):
            edges = edges.numpy()
        elif isinstance(bbox, o3d.geometry.OrientedBoundingBox):
            edges = np.asarray(edges)

        # project onto xy
        max_idxs = np.argmax(edges, axis=0)[:2]
        min_idxs = np.argmin(edges, axis=0)[:2]

        min_x_corner = edges[min_idxs[0]][:2]
        min_y_corner = edges[min_idxs[1]][:2]
        max_x_corner = edges[max_idxs[0]][:2]
        max_y_corner = edges[max_idxs[1]][:2]

        plt.plot([min_x_corner[0], max_y_corner[0]], [min_x_corner[1], max_y_corner[1]], "b-")
        plt.plot([min_x_corner[0], min_y_corner[0]], [min_x_corner[1], min_y_corner[1]], "b-")
        plt.plot([max_x_corner[0], max_y_corner[0]], [max_x_corner[1], max_y_corner[1]], "b-")
        plt.plot([min_y_corner[0], max_x_corner[0]], [min_y_corner[1], max_x_corner[1]], "b-")
        plt.annotate("Tile "+ str(i), xy=((min_x_corner[0] + max_x_corner[0])/2, (min_y_corner[1] + max_y_corner[1])/2))

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pointcloud", type=str, required=True)
    parser.add_argument("-s", "--size", type=int, required=True)
    parser.add_argument("-b", "--buffer", type=int)

    args = parser.parse_args()

    if not os.path.exists(args.pointcloud):
        print("couldnt find pointcloud")
        os._exit(1)
    
    tiled_pcs = tile_o3d_xy(args.pointcloud, size=args.size, buffer=args.buffer, exact_size=False, visualization=True)

    out_dir = "tiled/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i, pc in enumerate(tiled_pcs):
        o3d.io.write_point_cloud(out_dir+"Tile"+str(i)+".ply", pc)




# FOLLOWING: old o3d tiling, doesn't work


# def tileo3dsquare(pc: Union[str, o3d.geometry.PointCloud, o3d.t.geometry.PointCloud], size: int, buffer: int = None, out_dir: str = None, exact_size: bool = False, visualization: bool = False):
#     """
#     Tiles pointcloud (on x,y) into tiles of given size
#     The bool exact_size indicates wether exact given size is used: 
#         if True exact squares will be made and any left over parts will be smaller tiles.
#         if False the closest length to divide each side into exact squares will be used.
#     Parameters
#     ----------
#     pc
#         May be string with path to pointcloud, or o3d.(t) PointCloud object
#     size
#         Size of one side of tile in metres 
#     (optional) buffer
#         Buffer in meters to add to each tile border
#     (optional) exact_size
#         Wether to use exact given tile size
#     (optional) visualization
#         Visualization of tiles
#     """

#     # check type
#     if isinstance(pc, str):
#         if not os.path.exists(pc):
#             print(f"Can't find file at location {pc}")
#             return
#         pcd = o3d.io.read_point_cloud(pc)
#     elif isinstance(pc, o3d.geometry.PointCloud) or isinstance(pc, o3d.t.geometry.PointCloud):
#         pcd = pc
#     else:
#         print(f"Can't read pointcloud {pc}")
#         return
    
#     if size < 1:
#         print("size < 1")
#         return
    
    
#     oriented_bbox = pcd.get_oriented_bounding_box()
#     tiles = tile_o3dbbox(oriented_bbox, size, buffer, exact_size, visualization)

#     for i, tile in enumerate(tiles):
#         cropped = pcd.crop(tile)
        
#         # write output
#         if not out_dir:
#             out_dir = "tiled/"
#         if not os.path.exists(out_dir):
#             os.makedirs(out_dir)
#         o3d.io.write_point_cloud(out_dir+"Tile"+str(i)+".ply", cropped)

# def tile_o3dbbox(bbox: Union[o3d.geometry.OrientedBoundingBox, o3d.t.geometry.OrientedBoundingBox], size: int, buffer: int = None, exact_size: bool = False, visualization: bool = False):
#     """
#     Tiles given bbox into tiles of given size.
#     The bool exact_size indicates wether exact given size is used: 
#         if True exact squares will be made and any left over parts will be smaller tiles.
#         if False the closest length to divide each side into exact squares will be used.
#     Parameters
#     ----------
#     bbox
#         o3d.(t).geometry.OrientedBoundingBox
#     size
#         Size of one side of tile in metres 
#     (optional) buffer
#         Buffer in meters to add to each tile border
#     (optional) exact_size
#         Wether to use exact given tile size
#     (optional) visualization
#         Visualization of tiles
#     """

#     if isinstance(bbox, o3d.geometry.OrientedBoundingBox):
#         rot = np.copy(bbox.R)
#         rotation_center = np.copy(bbox.get_center())
#         aligned_bbox = bbox.rotate(rot.T)
#         Z_extent = aligned_bbox.extent[2]
#         aligned_center = aligned_bbox.get_center()
#         aligned_Z_max = aligned_center[2] + Z_extent/8
#         aligned_Z_min = aligned_center[2] - Z_extent/8
#     elif isinstance(bbox, o3d.t.geometry.OrientedBoundingBox):
#         rot = np.copy(bbox.rotation)
#         rotation_center = np.copy(bbox.GetCenter().numpy())
#         aligned_bbox = bbox.rotate(bbox.R.T())
#         Z_extent = aligned_bbox.extent
#     else:
#         print(f"Can't read bbox {bbox}")
#         return

#     # get corner points of rectangles
#     edges = aligned_bbox.get_box_points()
#     if isinstance(aligned_bbox, o3d.t.geometry.OrientedBoundingBox):
#         edges = edges.numpy()
#     elif isinstance(aligned_bbox, o3d.geometry.OrientedBoundingBox):
#         edges = np.asarray(edges)

#     tile_bboxs = create_tiles_aligned_corners(edges, size, buffer, exact_size)

#     rotated_bboxs = []

#     # for each tile: create o3d bbox and rotate back
#     for aligned_tile in tile_bboxs:
#         # first create 03d pointcloud, then get bounding box from this pointcloud
#         # no direct way of getting bbox in current o3d
#         top_corners = np.hstack((aligned_tile, np.asarray([[aligned_Z_max]]*4)))
#         bot_corners = np.hstack((aligned_tile, np.asarray([[aligned_Z_min]]*4)))

#         if isinstance(bbox, o3d.t.geometry.OrientedBoundingBox):
#             corners_pc = o3d.t.geometry.PointCloud(o3d.core.Tensor(np.vstack((top_corners, bot_corners))))
#         elif isinstance(bbox, o3d.geometry.OrientedBoundingBox):
#             corners_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack((top_corners, bot_corners))))
#         crop_bbox = corners_pc.get_oriented_bounding_box()


#         # rotate bbox back with big bbox center as rotation center
#         rotated = crop_bbox.rotate(rot, center=rotation_center)
#         rotated_bboxs.append(rotated)

#     if visualization:
#         visualize_bounds(rotated_bboxs)
#     return rotated_bboxs

