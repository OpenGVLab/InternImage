import os.path as osp
import os
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from shapely.geometry import LineString

def remove_nan_values(uv):
    is_u_valid = np.logical_not(np.isnan(uv[:, 0]))
    is_v_valid = np.logical_not(np.isnan(uv[:, 1]))
    is_uv_valid = np.logical_and(is_u_valid, is_v_valid)

    uv_valid = uv[is_uv_valid]
    return uv_valid

def points_ego2img(pts_ego, extrinsics, intrinsics):
    pts_ego_4d = np.concatenate([pts_ego, np.ones([len(pts_ego), 1])], axis=-1)
    pts_cam_4d = extrinsics @ pts_ego_4d.T
    
    uv = (intrinsics @ pts_cam_4d[:3, :]).T
    uv = remove_nan_values(uv)
    depth = uv[:, 2]
    uv = uv[:, :2] / uv[:, 2].reshape(-1, 1)

    return uv, depth

def interp_fixed_dist(line, sample_dist):
        ''' Interpolate a line at fixed interval.
        
        Args:
            line (LineString): line
            sample_dist (float): sample interval
        
        Returns:
            points (array): interpolated points, shape (N, 2)
        '''

        distances = list(np.arange(sample_dist, line.length, sample_dist))
        # make sure to sample at least two points when sample_dist > line.length
        distances = [0,] + distances + [line.length,] 
        
        sampled_points = np.array([list(line.interpolate(distance).coords)
                                for distance in distances]).squeeze()
        
        return sampled_points

def draw_polyline_ego_on_img(polyline_ego, img_bgr, extrinsics, intrinsics, color_bgr, thickness):
    # if 2-dimension, assume z=0
    if polyline_ego.shape[1] == 2:
        zeros = np.zeros((polyline_ego.shape[0], 1))
        polyline_ego = np.concatenate([polyline_ego, zeros], axis=1)

    polyline_ego = interp_fixed_dist(line=LineString(polyline_ego), sample_dist=0.2)
    
    uv, depth = points_ego2img(polyline_ego, extrinsics, intrinsics)

    h, w, c = img_bgr.shape

    is_valid_x = np.logical_and(0 <= uv[:, 0], uv[:, 0] < w - 1)
    is_valid_y = np.logical_and(0 <= uv[:, 1], uv[:, 1] < h - 1)
    is_valid_z = depth > 0
    is_valid_points = np.logical_and.reduce([is_valid_x, is_valid_y, is_valid_z])

    if is_valid_points.sum() == 0:
        return
    
    tmp_list = []
    for i, valid in enumerate(is_valid_points):
        
        if valid:
            tmp_list.append(uv[i])
        else:
            if len(tmp_list) >= 2:
                tmp_vector = np.stack(tmp_list)
                tmp_vector = np.round(tmp_vector).astype(np.int32)
                draw_visible_polyline_cv2(
                    copy.deepcopy(tmp_vector),
                    valid_pts_bool=np.ones((len(uv), 1), dtype=bool),
                    image=img_bgr,
                    color=color_bgr,
                    thickness_px=thickness,
                )
            tmp_list = []
    if len(tmp_list) >= 2:
        tmp_vector = np.stack(tmp_list)
        tmp_vector = np.round(tmp_vector).astype(np.int32)
        draw_visible_polyline_cv2(
            copy.deepcopy(tmp_vector),
            valid_pts_bool=np.ones((len(uv), 1), dtype=bool),
            image=img_bgr,
            color=color_bgr,
            thickness_px=thickness,
        )

    # uv = np.round(uv[is_valid_points]).astype(np.int32)
    # draw_visible_polyline_cv2(
    #     copy.deepcopy(uv),
    #     valid_pts_bool=np.ones((len(uv), 1), dtype=bool),
    #     image=img_bgr,
    #     color=color_bgr,
    #     thickness_px=thickness,
    # )

def draw_visible_polyline_cv2(line, valid_pts_bool, image, color, thickness_px):
    """Draw a polyline onto an image using given line segments.
    Args:
        line: Array of shape (K, 2) representing the coordinates of line.
        valid_pts_bool: Array of shape (K,) representing which polyline coordinates are valid for rendering.
            For example, if the coordinate is occluded, a user might specify that it is invalid.
            Line segments touching an invalid vertex will not be rendered.
        image: Array of shape (H, W, 3), representing a 3-channel BGR image
        color: Tuple of shape (3,) with a BGR format color
        thickness_px: thickness (in pixels) to use when rendering the polyline.
    """
    line = np.round(line).astype(int)  # type: ignore
    for i in range(len(line) - 1):

        if (not valid_pts_bool[i]) or (not valid_pts_bool[i + 1]):
            continue

        x1 = line[i][0]
        y1 = line[i][1]
        x2 = line[i + 1][0]
        y2 = line[i + 1][1]

        # Use anti-aliasing (AA) for curves
        image = cv2.line(image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=thickness_px, lineType=cv2.LINE_AA)


COLOR_MAPS_BGR = {
    # bgr colors
    'divider': (0, 0, 255),
    'boundary': (0, 255, 0),
    'ped_crossing': (255, 0, 0),
    'centerline': (51, 183, 255),
    'drivable_area': (171, 255, 255)
}

COLOR_MAPS_PLT = {
    'divider': 'r',
    'boundary': 'g',
    'ped_crossing': 'b',
    'centerline': 'orange',
    'drivable_area': 'y',
}

CAM_NAMES_AV2 = ['ring_front_center', 'ring_front_right', 'ring_front_left',
    'ring_rear_right','ring_rear_left', 'ring_side_right', 'ring_side_left',
]

class Renderer(object):
    """Render map elements on image views.
    Args:
        roi_size (tuple): bev range
    """

    def __init__(self, roi_size):
        self.roi_size = roi_size

    def render_bev_from_vectors(self, vectors, out_dir):
        '''Plot vectorized map elements on BEV.
        
        Args:
            vectors (dict): dict of vectorized map elements.
            out_dir (str): output directory
        '''

        car_img = Image.open('resources/images/car.png')
        map_path = os.path.join(out_dir, 'map.jpg')

        plt.figure(figsize=(self.roi_size[0], self.roi_size[1]))
        plt.xlim(-self.roi_size[0]/2 - 1, self.roi_size[0]/2 + 1)
        plt.ylim(-self.roi_size[1]/2 - 1, self.roi_size[1]/2 + 1)
        plt.axis('off')
        plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])

        for cat, vector_list in vectors.items():
            color = COLOR_MAPS_PLT[cat]
            for vector in vector_list:
                pts = np.array(vector)[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], angles='xy', color=color,
                #     scale_units='xy', scale=1)
                plt.plot(x, y, color=color, linewidth=5, marker='o', linestyle='-', markersize=20)

        plt.savefig(map_path, bbox_inches='tight', dpi=40)
        plt.close()
        
    def render_camera_views_from_vectors(self, vectors, imgs, extrinsics, 
            intrinsics, thickness, out_dir):
        '''Project vectorized map elements to camera views.
        
        Args:
            vectors (dict): dict of vectorized map elements.
            imgs (tensor): images in bgr color.
            extrinsics (array): ego2img extrinsics, shape (4, 4)
            intrinsics (array): intrinsics, shape (3, 3) 
            thickness (int): thickness of lines to draw on images.
            out_dir (str): output directory
        '''

        for i in range(len(imgs)):
            img = imgs[i]
            extrinsic = extrinsics[i]
            intrinsic = intrinsics[i]
            img_bgr = copy.deepcopy(img)

            for cat, vector_list in vectors.items():
                color = COLOR_MAPS_BGR[cat]
                for vector in vector_list:
                    img_bgr = np.ascontiguousarray(img_bgr)
                    vector_array = np.array(vector)
                    if vector_array.shape[1] > 3:
                        vector_array = vector_array[:, :3]
                    draw_polyline_ego_on_img(vector_array, img_bgr, extrinsic, intrinsic, 
                       color, thickness)

            out_path = osp.join(out_dir, CAM_NAMES_AV2[i]) + '.jpg'
            cv2.imwrite(out_path, img_bgr)
