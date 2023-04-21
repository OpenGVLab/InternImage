# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# openlane_v2_dataset.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
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

import os
import cv2
import torch
import numpy as np
from math import factorial
from pyquaternion import Quaternion

import mmcv
from mmdet.datasets import DATASETS
from mmdet3d.datasets import Custom3DDataset

from openlanev2.dataset import Collection
from openlanev2.evaluation import evaluate as openlanev2_evaluate
from openlanev2.preprocessing import check_results
from openlanev2.visualization.utils import COLOR_DICT


COLOR_GT = (0, 255, 0)
COLOR_GT_TOPOLOGY = (0, 127, 0)
COLOR_PRED = (0, 0, 255)
COLOR_PRED_TOPOLOGY = (0, 0, 127)
COLOR_DICT = {k: (v[2], v[1], v[0]) for k, v in COLOR_DICT.items()}


def render_pv(images, lidar2imgs, gt_lc, pred_lc, gt_te, gt_te_attr, pred_te, pred_te_attr):

    results = []

    for idx, (image, lidar2img) in enumerate(zip(images, lidar2imgs)):

        if gt_lc is not None :
            for lc in gt_lc:
                xyz1 = np.concatenate([lc, np.ones((lc.shape[0], 1))], axis=1)
                xyz1 = xyz1 @ lidar2img.T
                xyz1 = xyz1[xyz1[:, 2] > 1e-5]
                if xyz1.shape[0] == 0:
                    continue
                points_2d = xyz1[:, :2] / xyz1[:, 2:3]

                points_2d = points_2d.astype(int)
                image = cv2.polylines(image, points_2d[None], False, COLOR_GT, 2)

        if pred_lc is not None:
            for lc in pred_lc:
                xyz1 = np.concatenate([lc, np.ones((lc.shape[0], 1))], axis=1)
                xyz1 = xyz1 @ lidar2img.T
                xyz1 = xyz1[xyz1[:, 2] > 1e-5]
                if xyz1.shape[0] == 0:
                    continue
                points_2d = xyz1[:, :2] / xyz1[:, 2:3]

                points_2d = points_2d.astype(int)
                image = cv2.polylines(image, points_2d[None], False, COLOR_PRED, 2)

        if idx == 0: # front view image
            
            if gt_te is not None:
                for bbox, attr in zip(gt_te, gt_te_attr):
                    b = bbox.astype(np.int32)
                    image = render_corner_rectangle(image, (b[0], b[1]), (b[2], b[3]), COLOR_DICT[attr], 3, 1)

            if pred_te is not None:
                for bbox, attr in zip(pred_te, pred_te_attr):
                    b = bbox.astype(np.int32)
                    image = cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), COLOR_DICT[attr], 3)

        results.append(image)

    return results

def render_corner_rectangle(img, pt1, pt2, color,
                            corner_thickness=3, edge_thickness=2,
                            centre_cross=False, lineType=cv2.LINE_8):

    corner_length = min(abs(pt1[0] - pt2[0]), abs(pt1[1] - pt2[1])) // 4
    e_args = [color, edge_thickness, lineType]
    c_args = [color, corner_thickness, lineType]

    # edges
    img = cv2.line(img, (pt1[0] + corner_length, pt1[1]), (pt2[0] - corner_length, pt1[1]), *e_args)
    img = cv2.line(img, (pt2[0], pt1[1] + corner_length), (pt2[0], pt2[1] - corner_length), *e_args)
    img = cv2.line(img, (pt1[0], pt1[1] + corner_length), (pt1[0], pt2[1] - corner_length), *e_args)
    img = cv2.line(img, (pt1[0] + corner_length, pt2[1]), (pt2[0] - corner_length, pt2[1]), *e_args)

    # corners
    img = cv2.line(img, pt1, (pt1[0] + corner_length, pt1[1]), *c_args)
    img = cv2.line(img, pt1, (pt1[0], pt1[1] + corner_length), *c_args)
    img = cv2.line(img, (pt2[0], pt1[1]), (pt2[0] - corner_length, pt1[1]), *c_args)
    img = cv2.line(img, (pt2[0], pt1[1]), (pt2[0], pt1[1] + corner_length), *c_args)
    img = cv2.line(img, (pt1[0], pt2[1]), (pt1[0] + corner_length, pt2[1]), *c_args)
    img = cv2.line(img, (pt1[0], pt2[1]), (pt1[0], pt2[1] - corner_length), *c_args)
    img = cv2.line(img, pt2, (pt2[0] - corner_length, pt2[1]), *c_args)
    img = cv2.line(img, pt2, (pt2[0], pt2[1] - corner_length), *c_args)

    if centre_cross:
        cx, cy = int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2)
        img = cv2.line(img, (cx - corner_length, cy), (cx + corner_length, cy), *e_args)
        img = cv2.line(img, (cx, cy - corner_length), (cx, cy + corner_length), *e_args)
    
    return img

def render_front_view(image, lidar2img, gt_lc, pred_lc, gt_te, pred_te, gt_topology_lcte, pred_topology_lcte):

    if gt_topology_lcte is not None:
        for lc_idx, lcte in enumerate(gt_topology_lcte):
            for te_idx, connected in enumerate(lcte):
                if connected:
                    lc = gt_lc[lc_idx]
                    lc = lc[len(lc) // 2][None, ...]
                    xyz1 = np.concatenate([lc, np.ones((lc.shape[0], 1))], axis=1)
                    xyz1 = xyz1 @ lidar2img.T
                    xyz1 = xyz1[xyz1[:, 2] > 1e-5]
                    if xyz1.shape[0] == 0:
                        continue
                    p1 = (xyz1[:, :2] / xyz1[:, 2:3])[0].astype(int)

                    te = gt_te[te_idx]
                    p2 = np.array([(te[0]+te[2])/2, te[3]]).astype(int)

                    image = cv2.arrowedLine(image, (p2[0], p2[1]), (p1[0], p1[1]), COLOR_GT_TOPOLOGY, tipLength=0.03)

    if pred_topology_lcte is not None:
        for lc_idx, lcte in enumerate(pred_topology_lcte):
            for te_idx, connected in enumerate(lcte):
                if connected:
                    lc = pred_lc[lc_idx]
                    lc = lc[len(lc) // 2][None, ...]
                    xyz1 = np.concatenate([lc, np.ones((lc.shape[0], 1))], axis=1)
                    xyz1 = xyz1 @ lidar2img.T
                    xyz1 = xyz1[xyz1[:, 2] > 1e-5]
                    if xyz1.shape[0] == 0:
                        continue
                    p1 = (xyz1[:, :2] / xyz1[:, 2:3])[0].astype(int)

                    te = pred_te[te_idx]
                    p2 = np.array([(te[0]+te[2])/2, te[3]]).astype(int)

                    image = cv2.arrowedLine(image, (p2[0], p2[1]), (p1[0], p1[1]), COLOR_PRED_TOPOLOGY, tipLength=0.03)

    return image
    
def render_bev(gt_lc=None, pred_lc=None, gt_topology_lclc=None, pred_topology_lclc=None, map_size=[-52, 52, -27, 27], scale=20):

    image = np.zeros((int(scale*(map_size[1]-map_size[0])), int(scale*(map_size[3] - map_size[2])), 3), dtype=np.uint8)

    if gt_lc is not None:
        for lc in gt_lc:
            draw_coor = (scale * (-lc[:, :2] + np.array([map_size[1], map_size[3]]))).astype(np.int)
            image = cv2.polylines(image, [draw_coor[:, [1,0]]], False, COLOR_GT, max(round(scale * 0.2), 1))
            image = cv2.circle(image, (draw_coor[0, 1], draw_coor[0, 0]), max(round(scale * 0.5), 3), COLOR_GT, -1)
            image = cv2.circle(image, (draw_coor[-1, 1], draw_coor[-1, 0]), max(round(scale * 0.5), 3), COLOR_GT, -1)
    
    if gt_topology_lclc is not None:
        for l1_idx, lclc in enumerate(gt_topology_lclc):
            for l2_idx, connected in enumerate(lclc):
                if connected:
                    l1 = gt_lc[l1_idx]
                    l2 = gt_lc[l2_idx]
                    l1_mid = len(l1) // 2
                    l2_mid = len(l2) // 2
                    p1 = (scale * (-l1[l1_mid, :2] + np.array([map_size[1], map_size[3]]))).astype(np.int)
                    p2 = (scale * (-l2[l2_mid, :2] + np.array([map_size[1], map_size[3]]))).astype(np.int)
                    image = cv2.arrowedLine(image, (p1[1], p1[0]), (p2[1], p2[0]), COLOR_GT_TOPOLOGY, max(round(scale * 0.1), 1), tipLength=0.03)

    if pred_lc is not None:
        for lc in pred_lc:
            draw_coor = (scale * (-lc[:, :2] + np.array([map_size[1], map_size[3]]))).astype(np.int)
            image = cv2.polylines(image, [draw_coor[:, [1,0]]], False, COLOR_PRED, max(round(scale * 0.2), 1))
            image = cv2.circle(image, (draw_coor[0, 1], draw_coor[0, 0]), max(round(scale * 0.5), 3), COLOR_PRED, -1)
            image = cv2.circle(image, (draw_coor[-1, 1], draw_coor[-1, 0]), max(round(scale * 0.5), 3), COLOR_PRED, -1)

    if pred_topology_lclc is not None:
        for l1_idx, lclc in enumerate(pred_topology_lclc):
            for l2_idx, connected in enumerate(lclc):
                if connected:
                    l1 = pred_lc[l1_idx]
                    l2 = pred_lc[l2_idx]
                    l1_mid = len(l1) // 2
                    l2_mid = len(l2) // 2
                    p1 = (scale * (-l1[l1_mid, :2] + np.array([map_size[1], map_size[3]]))).astype(np.int)
                    p2 = (scale * (-l2[l2_mid, :2] + np.array([map_size[1], map_size[3]]))).astype(np.int)
                    image = cv2.arrowedLine(image, (p1[1], p1[0]), (p2[1], p2[0]), COLOR_PRED_TOPOLOGY, max(round(scale * 0.1), 1), tipLength=0.03)

    return image

@DATASETS.register_module()
class OpenLaneV2SubsetADataset(Custom3DDataset):

    CLASSES = [None]

    def __init__(self,
                 data_root,
                 meta_root,
                 collection,
                 pipeline,
                 test_mode,
                ):
        self.ann_file = f'{meta_root}/{collection}.pkl'
        super().__init__(
            data_root=data_root, 
            ann_file=self.ann_file, 
            pipeline=pipeline, 
            test_mode=test_mode,
        )

    def load_annotations(self, ann_file):
        ann_file = ann_file.name.split('.pkl')[0].split('/')
        self.collection = Collection(data_root=self.data_root, meta_root='/'.join(ann_file[:-1]), collection=ann_file[-1])
        return self.collection.keys

    def get_data_info(self, index):

        split, segment_id, timestamp = self.data_infos[index]
        frame = self.collection.get_frame_via_identifier((split, segment_id, timestamp))

        img_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        rots = []
        trans = []
        cam2imgs = []
        for i, camera in enumerate(frame.get_camera_list()):

            assert camera == 'ring_front_center' if i == 0 else True, \
                'the first image should be the front view'

            lidar2cam_r = np.linalg.inv(frame.get_extrinsic(camera)['rotation'])
            lidar2cam_t = frame.get_extrinsic(camera)['translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t

            intrinsic = frame.get_intrinsic(camera)['K']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)

            img_paths.append(frame.get_image_path(camera))
            lidar2cam_rts.append(lidar2cam_rt.T)
            cam_intrinsics.append(viewpad)
            lidar2img_rts.append(lidar2img_rt)
            rots.append(np.linalg.inv(frame.get_extrinsic(camera)['rotation']))
            trans.append(-frame.get_extrinsic(camera)['translation'])
            cam2imgs.append(frame.get_intrinsic(camera)['K'])

        can_bus = np.zeros(18)
        rotation = Quaternion._from_matrix(frame.get_pose()['rotation'])
        can_bus[:3] = frame.get_pose()['translation']
        can_bus[3:7] = rotation
        patch_angle = rotation.yaw_pitch_roll[0] / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        input_dict = {
            'scene_token': segment_id,
            'sample_idx': timestamp,
            'img_paths': img_paths,
            'lidar2cam': lidar2cam_rts,
            'cam_intrinsic': cam_intrinsics,
            'lidar2img': lidar2img_rts,
            'rots': rots,
            'trans': trans,
            'cam2imgs': cam2imgs,
            'can_bus': can_bus,
        }

        input_dict.update(self.get_ann_info(index))

        return input_dict

    def get_ann_info(self, index):

        split, segment_id, timestamp = self.data_infos[index]
        frame = self.collection.get_frame_via_identifier((split, segment_id, timestamp))

        gt_lc = np.array([lc['points'] for lc in frame.get_annotations_lane_centerlines()], dtype=np.float32)
        gt_lc_labels = np.zeros((len(gt_lc), ), dtype=np.int64)

        gt_te = np.array([element['points'].flatten() for element in frame.get_annotations_traffic_elements()], dtype=np.float32).reshape(-1, 4)
        gt_te_labels = np.array([element['attribute']for element in frame.get_annotations_traffic_elements()], dtype=np.int64)

        gt_topology_lclc = frame.get_annotations_topology_lclc()
        gt_topology_lcte = frame.get_annotations_topology_lcte()

        assert gt_lc.shape[0] == gt_topology_lclc.shape[0] == gt_topology_lclc.shape[1] == gt_topology_lcte.shape[0]
        assert gt_te.shape[0] == gt_topology_lcte.shape[1]

        return {
            'gt_lc': gt_lc,
            'gt_lc_labels': gt_lc_labels,
            'gt_te': gt_te,
            'gt_te_labels': gt_te_labels,
            'gt_topology_lclc': gt_topology_lclc,
            'gt_topology_lcte': gt_topology_lcte,
        }
    
    def pre_pipeline(self, results):
        pass

    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def evaluate(self, 
                 results, 
                 logger=None,
                 dump=None,
                 dump_dir=None,
                 visualization=False, 
                 visualization_dir=None,
                 visualization_num=None,
                 **kwargs):
        
        if logger:
            logger.info(f'Start formating...')
        pred_dict = self.format_preds(results)

        if dump:
            assert dump_dir is not None
            assert check_results(pred_dict), "Please fill the missing keys."
            output_path = os.path.join(dump_dir, 'result.pkl')
            mmcv.dump(pred_dict, output_path)

        if visualization:
            assert visualization_dir is not None
            self.visualize(pred_dict, visualization_dir, visualization_num, **kwargs)
        
        if logger:
            logger.info(f'Start evaluatation...')
        metric_results = {}
        for key, val in openlanev2_evaluate(ground_truth=self.ann_file, predictions=pred_dict).items():
            for k, v in val.items():
                metric_results[k if k != 'score' else key] = v
        return metric_results

    def format_preds(self, results):

        predictions = {
            'method': 'dummy',
            'authors': ['dummy'],
            'e-mail': 'dummy',
            'institution / company': 'dummy',
            # 'country / region': None,
            'results': {},
        }
        for index, result in enumerate(results):
            prediction = {                
                'lane_centerline': [],
                'traffic_element': [],
                'topology_lclc': None,
                'topology_lcte': None,
            }

            # lc

            pred_lc = result['pred_lc']
            sorted_index = np.argsort(pred_lc[1][:, 0])[:100]
            lanes, confidences = pred_lc[0][sorted_index], pred_lc[1][:, 0][sorted_index]

            lanes = lanes.reshape(-1, lanes.shape[-1] // 3, 3)

            def comb(n, k):
                return factorial(n) // (factorial(k) * factorial(n - k))
            n_points = 11
            n_control = lanes.shape[1]
            A = np.zeros((n_points, n_control))
            t = np.arange(n_points) / (n_points - 1)
            for i in range(n_points):
                for j in range(n_control):
                    A[i, j] = comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
            bezier_A = torch.tensor(A, dtype=torch.float32)
            lanes = torch.tensor(lanes, dtype=torch.float32)
            lanes = torch.einsum('ij,njk->nik', bezier_A, lanes)
            lanes = lanes.numpy()

            for i, (lane, confidence) in enumerate(zip(lanes, confidences)):
                prediction['lane_centerline'].append({
                    'id': i + 1000,
                    'points': lane.astype(np.float32),
                    'confidence': confidence,
                })

            # te

            pred_te = result['pred_te']
            for i, (bbox, confidence) in enumerate(zip(*pred_te)):
                prediction['traffic_element'].append({
                    'id': i + 2000,
                    'attribute': bbox[-1],
                    'points': bbox[:-1].reshape(2, 2).astype(np.float32),
                    'confidence': confidence,
                })

            # topology

            prediction['topology_lclc'] = result['pred_topology_lclc']
            prediction['topology_lcte'] = result['pred_topology_lcte']

            #

            predictions['results'][self.data_infos[index]] = {
                'predictions': prediction,
            }

        return predictions

    def visualize(self, pred_dict, visualization_dir, visualization_num, confidence_threshold=0.3, **kwargs):
        
        assert visualization_dir, 'Please specify visualization_dir for saving visualization.'

        print('\nStart visualization...\n')
            
        for index, (key, prediction) in enumerate(pred_dict['results'].items()):
            if visualization_num and index >= visualization_num:
                print(f'\nOnly {visualization_num} frames are visualized.\n')
                return

            frame = self.collection.get_frame_via_identifier(key)
            prediction = prediction['predictions']

            # calculate metric
            pred_result = {
                'method': 'dummy',
                'authors': 'dummy',
                'results': {
                    key: {
                        'predictions': prediction,
                    }
                }
            }
            gt_result = {key: {'annotation': frame.get_annotations()}}
            try:
                metric_results = openlanev2_evaluate(gt_result, pred_result, verbose=False)
            except Exception:
                metric_results = None

            # filter lc
            pred_lc_mask = np.array([lc['confidence'] for lc in prediction['lane_centerline']]) > confidence_threshold
            pred_lc = np.array([lc['points'] for lc in prediction['lane_centerline']])[pred_lc_mask]

            # filter te
            pred_te_mask = np.array([te['confidence'] for te in prediction['traffic_element']]) > confidence_threshold
            pred_te = np.array([te['points'].flatten() for te in prediction['traffic_element']])[pred_te_mask]
            pred_te_attr = np.array([te['attribute'] for te in prediction['traffic_element']])[pred_te_mask]

            # filter topology
            pred_topology_lclc = prediction['topology_lclc'][pred_lc_mask][:, pred_lc_mask] > confidence_threshold
            pred_topology_lcte = prediction['topology_lcte'][pred_lc_mask][:, pred_te_mask] > confidence_threshold
            
            data_info = self.get_data_info(index)
            if frame.get_annotations():
                gt_lc = np.array([lc['points'] for lc in frame.get_annotations_lane_centerlines()])

                gt_te = np.array([element['points'].flatten() for element in frame.get_annotations_traffic_elements()]).reshape(-1, 4)
                gt_te_attr = np.array([element['attribute']for element in frame.get_annotations_traffic_elements()])

                gt_topology_lclc = frame.get_annotations_topology_lclc()
                gt_topology_lcte = frame.get_annotations_topology_lcte()
            else:
                gt_lc, gt_te, gt_te_attr, gt_topology_lclc, gt_topology_lcte = None, None, None, None, None

            # render pv

            images = [mmcv.imread(img_path) for img_path in data_info['img_paths']]
            images = render_pv(
                images, data_info['lidar2img'], 
                gt_lc=gt_lc, pred_lc=pred_lc, 
                gt_te=gt_te, gt_te_attr=gt_te_attr, pred_te=pred_te, pred_te_attr=pred_te_attr,
            )
            for cam_idx, image in enumerate(images):
                output_path = os.path.join(visualization_dir, f'{"/".join(key)}/pv_{frame.get_camera_list()[cam_idx]}.jpg')
                mmcv.imwrite(image, output_path)

            img_pts = [
                (0, 3321, 2048, 4871),
                (356, 1273, 1906, 3321),
                (356, 4871, 1906, 6919),
                (2048, 4096, 3598, 6144),
                (2048, 2048, 3598, 4096),
                (2048, 6144, 3598, 8192),
                (2048, 0, 3598, 2048),
            ]
            multiview = np.zeros([3598, 8192, 3], dtype=np.uint8)
            for idx, pts in enumerate(img_pts):
                multiview[pts[0]:pts[2], pts[1]:pts[3]] = images[idx]
            multiview[2048:] = multiview[2048:, ::-1]
            multiview = cv2.resize(multiview, None, fx=0.5, fy=0.5)
            output_path = os.path.join(visualization_dir, f'{"/".join(key)}/pv_multiview.jpg')
            mmcv.imwrite(multiview, output_path)

            front_view = render_front_view(
                images[0], data_info['lidar2img'][0],
                gt_lc=gt_lc, pred_lc=pred_lc, 
                gt_te=gt_te, pred_te=pred_te,
                gt_topology_lcte=gt_topology_lcte,
                pred_topology_lcte=pred_topology_lcte,
            )
            output_path = os.path.join(visualization_dir, f'{"/".join(key)}/pv_{frame.get_camera_list()[0]}_topology.jpg')
            mmcv.imwrite(front_view, output_path)

            # render bev

            if metric_results is not None:
                info = []
                for k, v in metric_results['OpenLane-V2 Score'].items():
                    if k == 'score':
                        continue
                    info.append(f'{k}: {(lambda x: "%.2f" % x)(v)}')
                info = ' / '.join(info)
            else:
                info = '-'

            bev_lane = render_bev(
                gt_lc=gt_lc, pred_lc=pred_lc, 
                map_size=[-52, 55, -27, 27], scale=20,
            )
            bev_lane = cv2.putText(bev_lane, info, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GT, 2)
            output_path = os.path.join(visualization_dir, f'{"/".join(key)}/bev_lane.jpg')
            mmcv.imwrite(bev_lane, output_path)

            bev_gt = render_bev(
                gt_lc=gt_lc,
                gt_topology_lclc=gt_topology_lclc,
                map_size=[-52, 55, -27, 27], scale=20,
            )
            bev_pred = render_bev(
                pred_lc=pred_lc,  
                pred_topology_lclc=pred_topology_lclc,
                map_size=[-52, 55, -27, 27], scale=20,
            )
            divider = np.ones((bev_gt.shape[0], 7, 3), dtype=np.uint8) * 128
            bev_topology = np.concatenate([bev_gt, divider, bev_pred], axis=1)
            bev_topology = cv2.putText(bev_topology, info, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GT, 2)
            output_path = os.path.join(visualization_dir, f'{"/".join(key)}/bev_topology.jpg')
            mmcv.imwrite(bev_topology, output_path)
