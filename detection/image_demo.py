# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
import mmcv
import mmcv_custom  # noqa: F401,F403
import mmdet_custom  # noqa: F401,F403
import os.path as osp


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file or a directory contains images')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out', type=str, default="demo", help='out dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args

def inference_simgle_image(model, img_name, out_dir, score_thr, palette):
    # test a single image
    result = inference_detector(model, img_name)
    
    mmcv.mkdir_or_exist(out_dir)
    out_file = osp.join(out_dir, osp.basename(img_name))
    # show the results
    model.show_result(
        img_name,
        result,
        score_thr=score_thr,
        show=False,
        bbox_color=palette,
        text_color=(200, 200, 200),
        mask_color=palette,
        out_file=out_file
    )
    print(f"Result is save at {out_file}")


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    if osp.isdir(args.img):
        for img_name in mmcv.scandir(args.img, suffix=('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.webp')):
            inference_simgle_image(model, osp.join(args.img, img_name), args.out, args.score_thr, args.palette)
    else:
        inference_simgle_image(model, args.img, args.out, args.score_thr, args.palette)



if __name__ == '__main__':
    args = parse_args()
    main(args)