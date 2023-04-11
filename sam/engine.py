# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import time

import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F

import mmcv
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results


def prompt_sam_with_bboxes(sam_predictor, data, box_result):
    # process detector prediction
    # (x1, y1, x2, y2), rescaled in original image space
    bboxes = np.concatenate(box_result, axis=0)[..., :4]
    if len(bboxes) == 0:
        return [[] for _ in range(len(box_result))]
    labels = np.concatenate([[i] * len(boxes) for i, boxes in enumerate(box_result)])

    # prepare shapes
    img_metas = data['img_metas'][0].data[0][0]
    original_size = img_metas['ori_shape'][:2]

    # prepare input img of sam
    sam_predictor.reset_image()
    # img has been normed (NOTE 2.x norm img in pipeline)
    img = data['img'][0] .to(sam_predictor.model.device)
    # resize max length to 1024 and keep aspect ratio (ViT image encoder limitation)
    target_size = sam_predictor.transform.get_preprocess_shape(
        img.shape[2], img.shape[3],
        sam_predictor.transform.target_length)
    try:
        # `antialias=True` is provided in official implementation of SAM,
        # which may raise TypeError in PyTorch of previous versions.
        transformed_img = F.interpolate(
            img, target_size, mode="bilinear",
            align_corners=False, antialias=True)
    except TypeError:
        transformed_img = F.interpolate(
            img, target_size, mode="bilinear", align_corners=False)
    # Pad to 1024 x 1024
    h, w = transformed_img.shape[-2:]
    pad_h = sam_predictor.model.image_encoder.img_size - h
    pad_w = sam_predictor.model.image_encoder.img_size - w
    transformed_img = F.pad(transformed_img, (0, pad_w, 0, pad_h))

    # extract img feature
    sam_predictor.features = sam_predictor.model.image_encoder(
        transformed_img).to(sam_predictor.model.device)

    # set attributes
    sam_predictor.original_size = original_size
    sam_predictor.input_size = tuple(transformed_img.shape[-2:])
    sam_predictor.is_image_set = True

    # prepare bboxes and rescale bboxes to relative coordinates
    bboxes_tensor = torch.from_numpy(bboxes).to(sam_predictor.model.device)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(bboxes_tensor, original_size)

    # prompt with bboxes
    batch_masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False)

    batch_masks = batch_masks.squeeze(1).cpu().numpy()

    mask_results = [[*batch_masks[labels == i]] for i in range(len(box_result))]

    return mask_results


def single_gpu_test(model,
                    sam_predictor,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # For instance segmentor, only the box results is used in the
            # second stage (prompt sam with box). NOTE the mask_head is still
            # calculated, hence the FPS, FLOPS, maybe not accurate.
            result = model(return_loss=False, rescale=True, **data)
            if getattr(model.module, 'with_mask', False):
                box_result = result[0][0]  # simple_test supported
                mask_result = prompt_sam_with_bboxes(sam_predictor, data, box_result)
                result = [(box_result, mask_result)]
            else:
                raise NotImplementedError('WIP!')

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        # This logic is only used in panoptic segmentation test.
        elif isinstance(result[0], dict) and 'ins_results' in result[0]:
            for j in range(len(result)):
                bbox_results, mask_results = result[j]['ins_results']
                result[j]['ins_results'] = (bbox_results,
                                            encode_mask_results(mask_results))

        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results

