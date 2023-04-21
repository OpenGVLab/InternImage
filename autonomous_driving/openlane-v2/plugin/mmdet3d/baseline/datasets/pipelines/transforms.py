# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# transforms.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
#
# Contact wanghuijie@pjlab.org.cn if you have any issue.
#
# Copyright (c) 2023 The OpenLane-v2 Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
from numpy import random
from math import factorial

import mmcv
from mmdet.datasets import PIPELINES


@PIPELINES.register_module()
class ResizeFrontView:

    def __init__(self):
        pass

    def __call__(self, results):
        assert 'ring_front_center' in results['img_paths'][0], \
            'the first image should be the front view'

        #image
        front_view = results['img'][0]
        h, w, _ = front_view.shape
        resiezed_front_view, w_scale, h_scale = mmcv.imresize(
            front_view,
            (h, w),
            return_scale=True,
        )
        results['img'][0] = resiezed_front_view
        results['img_shape'][0] = resiezed_front_view.shape

        # gt
        scale_factor = np.array(
            [w_scale, h_scale, w_scale, h_scale],
            dtype=np.float32,
        )
        results['scale_factor'] = scale_factor
        if 'gt_te' in results:
            results['gt_te'] = results['gt_te'] * results['scale_factor']

        # intrinsic
        lidar2cam_r = results['rots'][0]
        lidar2cam_t = (-results['trans'][0]) @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t

        intrinsic = results['cam2imgs'][0]
        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

        cam_s = np.eye(4)
        cam_s[0, 0] *= w_scale
        cam_s[1, 1] *= h_scale

        viewpad = cam_s @ viewpad 
        intrinsic = viewpad[:intrinsic.shape[0], :intrinsic.shape[1]]
        lidar2img_rt = (viewpad @ lidar2cam_rt.T)

        results['cam_intrinsic'][0] = viewpad
        results['lidar2img'][0] = lidar2img_rt
        results['cam2imgs'][0] = intrinsic

        return results

@PIPELINES.register_module()
class NormalizeMultiviewImage:
    r"""
    Notes
    -----
    Adapted from https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py#L62.

    Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb


    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results['img'] = [mmcv.imnormalize(img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str

@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    r"""
    Notes
    -----
    Adapted from https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py#L99.
    
    Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str

@PIPELINES.register_module()
class CustomPadMultiViewImage:

    def __init__(self, size_divisor=None, pad_val=0):
        self.size_divisor = size_divisor
        self.pad_val = pad_val

    def __call__(self, results):
        max_h = max([img.shape[0] for img in results['img']])
        max_w = max([img.shape[1] for img in results['img']])
        padded_img = [mmcv.impad(img, shape=(max_h, max_w), pad_val=self.pad_val) for img in results['img']]
        if self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in padded_img]
        
        results['img'] = padded_img
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = None
        results['pad_size_divisor'] = self.size_divisor

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str

@PIPELINES.register_module()
class CustomParameterizeLane:

    def __init__(self, method, method_para):
        method_list = ['bezier', 'polygon', 'bezier_Direction_attribute', 'bezier_Endpointfixed']
        self.method = method
        if not self.method in method_list:
            raise Exception("Not implemented!")
        self.method_para = method_para

    def __call__(self, results):
        centerlines = results['gt_lc']
        para_centerlines = getattr(self, self.method)(centerlines, **self.method_para)
        results['gt_lc'] = para_centerlines
        return results

    def comb(self, n, k):
        return factorial(n) // (factorial(k) * factorial(n - k))

    def fit_bezier(self, points, n_control):
        n_points = len(points)
        A = np.zeros((n_points, n_control))
        t = np.arange(n_points) / (n_points - 1)
        for i in range(n_points):
            for j in range(n_control):
                A[i, j] = self.comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
        conts = np.linalg.lstsq(A, points, rcond=None)
        return conts

    def fit_bezier_Endpointfixed(self, points, n_control):
        n_points = len(points)
        A = np.zeros((n_points, n_control))
        t = np.arange(n_points) / (n_points - 1)
        for i in range(n_points):
            for j in range(n_control):
                A[i, j] = self.comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
        A_BE = A[1:-1, 1:-1]
        _points = points[1:-1]
        _points = _points - A[1:-1, 0].reshape(-1, 1) @ points[0].reshape(1, -1) - A[1:-1, -1].reshape(-1, 1) @ points[-1].reshape(1, -1)

        conts = np.linalg.lstsq(A_BE, _points, rcond=None)

        control_points = np.zeros((n_control, points.shape[1]))
        control_points[0] = points[0]
        control_points[-1] = points[-1]
        control_points[1:-1] = conts[0]

        return control_points

    def bezier(self, input_data, n_control=2):

        coeffs_list = []
        for idx, centerline in enumerate(input_data):
            sorted_x = np.array(centerline[:, 1])
            sorted_y = np.array(centerline[:, 0])
            points = np.array(list(zip(sorted_x, sorted_y)))
            res = self.fit_bezier(points, n_control)[0]
            start_res = res[0]
            end_res = res[-1]
            first_diff = (np.sum(np.square(start_res - points[0]))) + (np.sum(np.square(end_res - points[-1])))
            second_diff = (np.sum(np.square(start_res - points[-1]))) + (np.sum(np.square(end_res - points[0])))

            if first_diff <= second_diff:
                fin_res = res
            else:
                fin_res = np.zeros_like(res)
                for m in range(len(res)):
                    fin_res[len(res) - m - 1] = res[m]

            fin_res = np.clip(fin_res, 0, 1)
            coeffs_list.append(np.reshape(np.float32(fin_res), (-1)))

        return np.array(coeffs_list)

    def bezier_Direction_attribute(self, input_data, n_control=3):
        coeffs_list = []
        for idx, centerline in enumerate(input_data):
            centerline[:, 1] = centerline[:, 1]
            centerline[:, 0] = centerline[:, 0]
            sorted_x = np.array(centerline[:, 1])
            sorted_y = np.array(centerline[:, 0])
            points = np.array(list(zip(sorted_x, sorted_y)))
            res = self.fit_bezier(points, n_control)[0]
            fin_res = np.clip(res, 0, 1)
            start_res = res[0]
            end_res = res[-1]
            first_diff = (np.sum(np.square(start_res - points[0]))) + (np.sum(np.square(end_res - points[-1])))
            second_diff = (np.sum(np.square(start_res - points[-1]))) + (np.sum(np.square(end_res - points[0])))
            if first_diff <= second_diff:
                da = 0
            else:
                da = 1
            fin_res = np.append(fin_res, da)
            coeffs_list.append(np.reshape(np.float32(fin_res), (-1)))
        return np.array(coeffs_list)

    def bezier_Endpointfixed(self, input_data, n_control=2):
        coeffs_list = []
        for idx, centerline in enumerate(input_data):
            res = self.fit_bezier_Endpointfixed(centerline, n_control)
            coeffs = res.flatten()
            coeffs_list.append(coeffs)
        return np.array(coeffs_list, dtype=np.float32)

    def polygon(self, input_data, key_rep='Bounding Box'):
        keypoints = []
        for idx, centerline in enumerate(input_data):
            centerline[:, 1] = centerline[:, 1]
            centerline[:, 0] = centerline[:, 0]
            sorted_x = np.array(centerline[:, 1])
            sorted_y = np.array(centerline[:, 0])
            points = np.array(list(zip(sorted_x, sorted_y)))
            if key_rep not in ['Bounding Box', 'SME', 'Extreme Points']:
                raise Exception(f"{key_rep} not existed!")
            elif key_rep == 'Bounding Box':
                res = np.array(
                    [points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()]).reshape((2, 2))
                keypoints.append(np.reshape(np.float32(res), (-1)))
            elif key_rep == 'SME':
                res = np.array([points[0], points[-1], points[int(len(points) / 2)]])
                keypoints.append(np.reshape(np.float32(res), (-1)))
            else:
                min_x = np.min([points[:, 0] for p in points])
                ind_left = np.where(points[:, 0] == min_x)
                max_x = np.max([points[:, 0] for p in points])
                ind_right = np.where(points[:, 0] == max_x)
                max_y = np.max([points[:, 1] for p in points])
                ind_top = np.where(points[:, 1] == max_y)
                min_y = np.min([points[:, 1] for p in points])
                ind_botton = np.where(points[:, 1] == min_y)
                res = np.array(
                    [points[ind_left[0][0]], points[ind_right[0][0]], points[ind_top[0][0]], points[ind_botton[0][0]]])
                keypoints.append(np.reshape(np.float32(res), (-1)))
        return np.array(keypoints)
