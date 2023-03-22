import argparse
import os
import pickle as pkl
import numpy as np
import random
from PIL import Image
import concurrent.futures
import json
import mmcv

def parse_args():
    parser = argparse.ArgumentParser(description='Generate MMDetection Annotations for Crowdhuman-like dataset')
    parser.add_argument('--dataset', help='dataset name', type=str)
    parser.add_argument('--dataset-split', help='dataset split, e.g. train, val', type=str)

    args = parser.parse_args()
    return args.dataset, args.dataset_split

def load_func(fpath):
    assert os.path.exists(fpath)
    with open(fpath, 'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]
    return records

def decode_annotations(records, dataset_path):
    rec_ids = list(range(len(records)))
    img_list = []
    ann_list = []
    ann_id = 1
    for idx, rec_id in enumerate(rec_ids):
        img_id = records[rec_id]['ID']
        img_url = dataset_path + 'Images/' + img_id + '.jpg'
        assert os.path.exists(img_url)
        im = Image.open(img_url)
        im_w, im_h = im.width, im.height

        gt_box = records[rec_id]['gtboxes']
        gt_box_len = len(gt_box)
        img_dict = dict(
            file_name=img_id + '.jpg',
            height=im_h,
            width=im_w,
            id=idx
        )
        img_list.append(img_dict)
        for ii in range(gt_box_len):
            each_data = gt_box[ii]
            x, y, w, h = each_data['fbox']

            if w <= 0 or h <= 0:
                continue
            # x1 = x; y1 = y; x2 = x + w; y2 = y + h

            valid_bbox = [x, y, w, h]
            if each_data['tag'] == 'person':
                tag = 1
            else:
                tag = -2
            if 'extra' in each_data:
                if 'ignore' in each_data['extra']:
                    if each_data['extra']['ignore'] != 0:
                        tag = -2
            ann_dict = dict(
                area=w * h,
                iscrowd=1 if tag == -2 else 0,
                image_id=idx,
                bbox=[x, y, w, h],
                category_id=1,
                id=ann_id,
                # ignore=1 if tag == -2 else 1,
            )
            ann_id += 1
            ann_list.append(ann_dict)
    cate_list = [{'supercategory': 'none', 'id': 1, 'name': 'person'}]
    json_dict = dict(
        images=img_list,
        annotations=ann_list,
        categories=cate_list
    )
    return json_dict

if __name__ == "__main__":
    dataset_name, dataset_type = parse_args()
    dataset_path = 'data/%s/' % dataset_name
    ch_file_path = dataset_path + 'annotations/annotation_%s.odgt' % dataset_type
    json_file_path = dataset_path + 'annotations/annotation_%s.json' % dataset_type

    records = load_func(ch_file_path)
    print("Loading Annotations Done")

    json_dict = decode_annotations(records, dataset_path)

    print("Parsing Bbox Number: %d" % len(json_dict['annotations']))
    mmcv.dump(json_dict, json_file_path)
