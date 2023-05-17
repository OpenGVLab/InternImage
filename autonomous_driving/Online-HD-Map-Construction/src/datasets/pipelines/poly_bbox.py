import numpy as np

from mmdet.datasets.builder import PIPELINES
from shapely.geometry import LineString

@PIPELINES.register_module(force=True)
class PolygonizeLocalMapBbox(object):
    """Pre-Processing used by vectormapnet model.

    Args:
        canvas_size (tuple or list): bev feature size
        coord_dim (int): dimension of point's coordinate
        num_class (int): number of classes
        threshold (float): threshold for minimum bounding box size
    """

    def __init__(self,
                 canvas_size=(200, 100),
                 coord_dim=2,
                 num_class=3,
                 threshold=6/200,
                 ):

        self.canvas_size = np.array(canvas_size)

        self.num_class = num_class

        # for keypoints
        self.threshold = threshold

        self.coord_dim = coord_dim

        self.map_stop_idx = 0
        self.coord_dim_start_idx = 1

    def format_polyline_map(self, vectors):

        polylines, polyline_masks, polyline_weights = [], [], []

        # quantilize each label's lines individually.
        for label, _lines in vectors.items():
            for polyline in _lines:
                # and pad polyline.
                if label == 2:
                    polyline_weight = evaluate_line(polyline).reshape(-1)
                else:
                    polyline_weight = np.ones_like(polyline).reshape(-1)
                    polyline_weight = np.pad(
                        polyline_weight, ((0, 1),), constant_values=1.)
                    polyline_weight = polyline_weight/polyline_weight.sum()

                # flatten and quantilized
                fpolyline = quantize_verts(
                    polyline, self.canvas_size, self.coord_dim)

                fpolyline = fpolyline.reshape(-1)

                # reindex starting from 1, and add a zero stopping token(EOS),
                fpolyline = \
                    np.pad(fpolyline + self.coord_dim_start_idx, ((0, 1),),
                            constant_values=0)
                fpolyline_msk = np.ones(fpolyline.shape, dtype=np.bool)

                polyline_masks.append(fpolyline_msk)
                polyline_weights.append(polyline_weight)
                polylines.append(fpolyline)

        polyline_map = polylines
        polyline_map_mask = polyline_masks
        polyline_map_weights = polyline_weights

        return polyline_map, polyline_map_mask, polyline_map_weights

    def format_keypoint(self, vectors):

        kps, kp_labels = [], []
        qkps, qkp_masks = [], []

        # quantilize each label's lines individually.
        for label, _lines in vectors.items():
            for polyline in _lines:
                kp = get_bbox(polyline, self.threshold)
                kps.append(kp)
                kp_labels.append(label)

                gkp = kp

                # flatten and quantilized
                fkp = quantize_verts(gkp, self.canvas_size, self.coord_dim)
                fkp = fkp.reshape(-1)

                fkps_msk = np.ones(fkp.shape, dtype=np.bool)

                qkp_masks.append(fkps_msk)
                qkps.append(fkp)

        qkps = np.stack(qkps)
        qkp_msks = np.stack(qkp_masks)

        # format det
        kps = np.stack(kps, axis=0).astype(np.float32)*self.canvas_size
        kp_labels = np.array(kp_labels)
        # restrict the boundary
        kps[..., 0] = np.clip(kps[..., 0], 0.1, self.canvas_size[0]-0.1)
        kps[..., 1] = np.clip(kps[..., 1], 0.1, self.canvas_size[1]-0.1)

        # nbox, boxsize(4)*coord_dim(2)
        kps = kps.reshape(kps.shape[0], -1)
        # unflatten_seq(qkps)

        return kps, kp_labels, qkps, qkp_msks,

    def Polygonization(self, input_dict):
        '''
            Process vertices.
        '''
        
        vectors = input_dict['vectors']

        n_lines = 0
        for label, lines in vectors.items():
            n_lines += len(lines)
        if not n_lines:
            input_dict['polys'] = []
            return input_dict

        polyline_map, polyline_map_mask, polyline_map_weight = \
            self.format_polyline_map(vectors)

        keypoint, keypoint_label, qkeypoint, qkeypoint_mask = \
            self.format_keypoint(vectors)

        # gather
        polys = {
            # for det
            'keypoint': keypoint,
            'det_label': keypoint_label,

            # for gen
            'gen_label': keypoint_label,
            'qkeypoint': qkeypoint,
            'qkeypoint_mask': qkeypoint_mask,

            'polylines': polyline_map,  # List[array]
            'polyline_masks': polyline_map_mask,  # List[array]
            'polyline_weights': polyline_map_weight
        }

        # Format outputs
        input_dict['polys'] = polys

        return input_dict

    def __call__(self, input_dict):
        input_dict = self.Polygonization(input_dict)
        return input_dict


def evaluate_line(polyline):

    edge = np.linalg.norm(polyline[1:] - polyline[:-1], axis=-1)

    start_end_weight = edge[(0, -1), ].copy()
    mid_weight = (edge[:-1] + edge[1:]) * .5

    pts_weight = np.concatenate(
        (start_end_weight[:1], mid_weight, start_end_weight[-1:]))

    denominator = pts_weight.sum()
    denominator = 1 if denominator == 0 else denominator

    pts_weight /= denominator

    # add weights for stop index
    pts_weight = np.repeat(pts_weight, 2)/2
    pts_weight = np.pad(pts_weight, ((0, 1)),
                        constant_values=1/(len(polyline)*2))

    return pts_weight


def quantize_verts(verts, canvas_size, coord_dim):
    """Convert vertices from its original range ([-1,1]) to discrete values in [0, n_bits**2 - 1].
    
    Args:
        verts (array): vertices coordinates, shape (seqlen, coords_dim)
        canvas_size (tuple): bev feature size
        coord_dim (int): dimension of point coordinates

    Returns:
        quantized_verts (array): quantized vertices, shape (seqlen, coords_dim)
    """

    min_range = 0
    max_range = 1
    range_quantize = np.array(canvas_size) - 1  # (0-199) = 200

    verts_ratio = (verts[:, :coord_dim] - min_range) / (
        max_range - min_range)
    verts_quantize = verts_ratio * range_quantize[:coord_dim]

    return verts_quantize.astype('int32')


def get_bbox(polyline, threshold):
    """Convert vertices from its original range ([-1,1]) to discrete values in [0, n_bits**2 - 1].
    
    Args:
        polyline (array): point coordinates, shape (seqlen, 2)
        threshold (float): threshold for minimum bbox size
    
    Returns:
        bbox (array): bounding box in xyxy format, shape (2, 2)
    """
    eps = 1e-4
    polyline = LineString(polyline)
    bbox = polyline.bounds
    minx, miny, maxx, maxy = bbox
    W, H = maxx-minx, maxy-miny

    if W < threshold or H < threshold:
        remain = max((threshold - min(W, H))/2, eps)
        bbox = polyline.buffer(remain).envelope.bounds
        minx, miny, maxx, maxy = bbox

    bbox_np = np.array([[minx, miny], [maxx, maxy]])
    bbox_np = np.clip(bbox_np, 0., 1.)

    return bbox_np