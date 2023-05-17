import argparse
import mmcv
from mmcv import Config
import os
from renderer import Renderer

CAT2ID = {
    'ped_crossing': 0,
    'divider': 1,
    'boundary': 2,
}

ID2CAT = {v: k for k, v in CAT2ID.items()}

ROI_SIZE = (60, 30)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize groundtruth and results')
    
    parser.add_argument('log_id', type=str,
        help='log_id of data to visualize')
    parser.add_argument('ann_file', 
        help='gt file to visualize')
    parser.add_argument('--result', 
        type=str,
        help='prediction result to visualize')
    parser.add_argument('--thr', 
        type=float,
        default=0,
        help='score threshold to filter predictions')
    parser.add_argument(
        '--out-dir', 
        default='demo',
        help='directory where visualize results will be saved')
    args = parser.parse_args()

    return args

def import_plugin(cfg):
    '''
        import modules, registry will be update
    '''

    import sys
    sys.path.append(os.path.abspath('.'))    
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                def import_path(plugin_dir):
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]

                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(f'importing {_module_path}/')
                    plg_lib = importlib.import_module(_module_path)

                plugin_dirs = cfg.plugin_dir
                if not isinstance(plugin_dirs,list):
                    plugin_dirs = [plugin_dirs,]
                for plugin_dir in plugin_dirs:
                    import_path(plugin_dir)
                
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(f'importing {_module_path}/')
                plg_lib = importlib.import_module(_module_path)

def main(args):
    log_id = args.log_id
    ann = mmcv.load(args.ann_file)
    root_path = os.path.dirname(args.ann_file)
    out_dir = os.path.join(args.out_dir, str(log_id))
    
    log_ann = ann[log_id]
    renderer = Renderer(roi_size=ROI_SIZE)

    if args.result:
        result = mmcv.load(args.result)['results']

    for frame in mmcv.track_iter_progress(log_ann):
        timestamp = frame['timestamp']
        sensor = frame['sensor']
        annotation = frame['annotation']
        imgs = [mmcv.imread(os.path.join(root_path, 'argoverse2', i['image_path'])) for i in sensor.values()]
        extrinsics = [i['extrinsic'] for i in sensor.values()]
        intrinsics = [i['intrinsic'] for i in sensor.values()]

        frame_dir = os.path.join(out_dir, timestamp, 'gt')
        os.makedirs(frame_dir, exist_ok=True)
        
        renderer.render_bev_from_vectors(annotation, out_dir=frame_dir)
        renderer.render_camera_views_from_vectors(annotation, imgs, extrinsics, 
            intrinsics, 4, frame_dir)

        if args.result:
            pred = result[timestamp]
            vectors = {cat: [] for cat in CAT2ID.keys()}
            for i in range(len(pred['labels'])):
                score = pred['scores'][i]
                label = pred['labels'][i]
                v = pred['vectors'][i]

                if score > args.thr:
                    vectors[ID2CAT[label]].append(v)
            
            frame_dir = os.path.join(out_dir, timestamp, 'pred')
            os.makedirs(frame_dir, exist_ok=True)
            renderer.render_bev_from_vectors(vectors, out_dir=frame_dir)
            renderer.render_camera_views_from_vectors(vectors, imgs, 
                    extrinsics, intrinsics, 4, frame_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)