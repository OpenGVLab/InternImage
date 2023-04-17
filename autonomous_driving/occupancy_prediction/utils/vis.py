import open3d as o3d
import pickle
import numpy as np
import torch
import math
from pathlib import Path
import os
from glob import glob

LINE_SEGMENTS = [
    [4, 0], [3, 7], [5, 1], [6, 2],  # lines along x-axis
    [5, 4], [5, 6], [6, 7], [7, 4],  # lines along x-axis
    [0, 1], [1, 2], [2, 3], [3, 0]]  # lines along y-axis
colors_map = np.array(
    [
        # [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
    ])
color = colors_map[:, :3] / 255


def voxel2points(voxel, voxelSize, range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4], ignore_labels=[17, 255]):
    if isinstance(voxel, np.ndarray): voxel = torch.from_numpy(voxel)
    mask = torch.zeros_like(voxel, dtype=torch.bool)
    for ignore_label in ignore_labels:
        mask = torch.logical_or(voxel == ignore_label, mask)
    mask = torch.logical_not(mask)
    occIdx = torch.where(mask)
    # points = torch.concatenate((np.expand_dims(occIdx[0], axis=1) * voxelSize[0], \
    #                          np.expand_dims(occIdx[1], axis=1) * voxelSize[1], \
    #                          np.expand_dims(occIdx[2], axis=1) * voxelSize[2]), axis=1)
    points = torch.cat((occIdx[0][:, None] * voxelSize[0] + voxelSize[0] / 2 + range[0], \
                        occIdx[1][:, None] * voxelSize[1] + voxelSize[1] / 2 + range[1], \
                        occIdx[2][:, None] * voxelSize[2] + voxelSize[2] / 2 + range[2]), dim=1)
    return points, voxel[occIdx]

def voxel_profile(voxel, voxel_size):
    centers = torch.cat((voxel[:, :2], voxel[:, 2][:, None] - voxel_size[2] / 2), dim=1)
    # centers = voxel
    wlh = torch.cat((torch.tensor(voxel_size[0]).repeat(centers.shape[0])[:, None],
                          torch.tensor(voxel_size[1]).repeat(centers.shape[0])[:, None],
                          torch.tensor(voxel_size[2]).repeat(centers.shape[0])[:, None]), dim=1)
    yaw = torch.full_like(centers[:, 0:1], 0)
    return torch.cat((centers, wlh, yaw), dim=1)

def rotz(t):
    """Rotation about the z-axis."""
    c = torch.cos(t)
    s = torch.sin(t)
    return torch.tensor([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def my_compute_box_3d(center, size, heading_angle):
    h, w, l = size[:, 2], size[:, 0], size[:, 1]
    heading_angle = -heading_angle - math.pi / 2
    center[:, 2] = center[:, 2] + h / 2
    #R = rotz(1 * heading_angle)
    l, w, h = (l / 2).unsqueeze(1), (w / 2).unsqueeze(1), (h / 2).unsqueeze(1)
    x_corners = torch.cat([-l, l, l, -l, -l, l, l, -l], dim=1)[..., None]
    y_corners = torch.cat([w, w, -w, -w, w, w, -w, -w], dim=1)[..., None]
    z_corners = torch.cat([h, h, h, h, -h, -h, -h, -h], dim=1)[..., None]
    #corners_3d = R @ torch.vstack([x_corners, y_corners, z_corners])
    corners_3d = torch.cat([x_corners, y_corners, z_corners], dim=2)
    corners_3d[..., 0] += center[:, 0:1]
    corners_3d[..., 1] += center[:, 1:2]
    corners_3d[..., 2] += center[:, 2:3]
    return corners_3d

def generate_the_ego_car():
    ego_range = [-2, -1, 0, 2, 1, 1.5]
    ego_voxel_size=[0.1, 0.1, 0.1]
    ego_xdim = int((ego_range[3] - ego_range[0]) / ego_voxel_size[0])
    ego_ydim = int((ego_range[4] - ego_range[1]) / ego_voxel_size[1])
    ego_zdim = int((ego_range[5] - ego_range[2]) / ego_voxel_size[2])
    ego_voxel_num = ego_xdim * ego_ydim * ego_zdim
    temp_x = np.arange(ego_xdim)
    temp_y = np.arange(ego_ydim)
    temp_z = np.arange(ego_zdim)
    ego_xyz = np.stack(np.meshgrid(temp_y, temp_x, temp_z), axis=-1).reshape(-1, 3)
    ego_point_x = (ego_xyz[:, 0:1] + 0.5) / ego_xdim * (ego_range[3] - ego_range[0]) + ego_range[0]
    ego_point_y = (ego_xyz[:, 1:2] + 0.5) / ego_ydim * (ego_range[4] - ego_range[1]) + ego_range[1]
    ego_point_z = (ego_xyz[:, 2:3] + 0.5) / ego_zdim * (ego_range[5] - ego_range[2]) + ego_range[2]
    ego_point_xyz = np.concatenate((ego_point_y, ego_point_x, ego_point_z), axis=-1)
    ego_points_label =  (np.ones((ego_point_xyz.shape[0]))*16).astype(np.uint8)
    ego_dict = {}
    ego_dict['point'] = ego_point_xyz
    ego_dict['label'] = ego_points_label
    return ego_point_xyz

def show_point_cloud(points: np.ndarray, colors=True, points_colors=None, obj_bboxes=None, voxelize=False, bbox_corners=None, linesets=None, ego_pcd=None, scene_idx=0, frame_idx=0, large_voxel=True, voxel_size=0.4) -> None:
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(str(scene_idx))

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(points_colors[:, :3])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.6, origin=[0, 0, 0])

    pcd.points = o3d.utility.Vector3dVector(points)
    voxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    if large_voxel:
        vis.add_geometry(voxelGrid)
    else:
        vis.add_geometry(pcd)
    if voxelize:
        line_sets = o3d.geometry.LineSet()
        line_sets.points = o3d.open3d.utility.Vector3dVector(bbox_corners.reshape((-1, 3)))
        line_sets.lines = o3d.open3d.utility.Vector2iVector(linesets.reshape((-1, 2)))
        line_sets.paint_uniform_color((0, 0, 0))

    vis.add_geometry(mesh_frame)
    vis.add_geometry(pcd)
    view_control = vis.get_view_control()
    view_control.set_lookat(np.array([0, 0, 0]))
    vis.add_geometry(line_sets)
    vis.poll_events()
    vis.update_renderer()
    return vis

def vis_nuscene():
    voxelSize = [0.4, 0.4, 0.4]
    point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]

    ignore_labels = [17, 255]
    vis_voxel_size = 0.4
    file = "data/29796060110c4163b07f06eff4af0753/labels.npz"
    data = np.load(file)

    semantics, mask_lidar, mask_camera = data['semantics'], data['mask_lidar'], data['mask_camera']
    voxels = semantics

    points, labels = voxel2points(voxels, voxelSize, range=point_cloud_range, ignore_labels=ignore_labels)
    points = points.numpy()
    labels = labels.numpy()
    pcd_colors = color[labels.astype(int) % len(color)]
    bboxes = voxel_profile(torch.tensor(points), voxelSize)
    ego_pcd = o3d.geometry.PointCloud()
    ego_points = generate_the_ego_car()
    ego_pcd.points = o3d.utility.Vector3dVector(ego_points)
    bboxes_corners = my_compute_box_3d(bboxes[:, 0:3], bboxes[:, 3:6], bboxes[:, 6:7])
    bases_ = torch.arange(0, bboxes_corners.shape[0] * 8, 8)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])  # lines along y-axis
    edges = edges.reshape((1, 12, 2)).repeat(bboxes_corners.shape[0], 1, 1)
    edges = edges + bases_[:, None, None]
    vis = show_point_cloud(points=points, colors=True, points_colors=pcd_colors, voxelize=True, obj_bboxes=None,
                        bbox_corners=bboxes_corners.numpy(), linesets=edges.numpy(), ego_pcd=ego_pcd, large_voxel=True, voxel_size=vis_voxel_size)

    # control view    
    # view_control = vis.get_view_control()
    # view_control.set_zoom(args.zoom)
    # view_control.set_up(args.up_vec)
    # view_control.set_front(args.front_vec)
    # view_control.set_lookat(np.array([points.mean(axis=0)[0], 0, 0]))
    # vis.poll_events()
    # vis.update_renderer()

    vis.run()
    # vis.capture_screen_image(os.path.join(images_outdir, "{}.png".format(file_name)))

    vis.destroy_window()
    del vis

if __name__ == '__main__':
    vis_nuscene()