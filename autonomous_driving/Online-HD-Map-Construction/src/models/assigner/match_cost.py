import torch
from mmdet.core.bbox.match_costs.builder import MATCH_COST
from mmdet.core.bbox.match_costs import build_match_cost

from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy


def chamfer_distance(pred, gt):
    '''
    Args:
    pred: [num_points, 2]
    gt: [num_gt, 2]
    Out: torch.FloatTensor of shape (1, )
    '''
    # [num_points, num_gt]
    dist_mat = torch.cdist(pred, gt, p=2)
    # [num_points]
    dist_pred, _ = torch.min(dist_mat, dim=-1)

    dist_pred = torch.clamp(dist_pred, max=2.0)

    dist_pred = dist_pred.mean()

    dist_gt, _ = torch.min(dist_mat, dim=0)
    dist_gt = torch.clamp(dist_gt, max=2.0)
    dist_gt = dist_gt.mean()

    dist = dist_pred + dist_gt
    return dist


@MATCH_COST.register_module()
class ClsSigmoidCost:
    """ClsSoftmaxCost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
        Returns:
            torch.Tensor: cls_cost value with weight
        """
        # Following the official DETR repo, contrary to the loss that
        # NLL is used, we approximate it in 1 - cls_score[gt_label].
        # The 1 is a constant that doesn't change the matching,
        # so it can be omitted.
        cls_score = cls_pred.sigmoid()
        cls_cost = -cls_score[:, gt_labels]
        return cls_cost * self.weight


@MATCH_COST.register_module()
class LinesFixNumChamferCost(object):
    """BBox3DL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, lines_pred, gt_lines):
        """
        Args:
            lines_pred (Tensor): predicted normalized lines:
                [num_query, num_points, 2]
            gt_lines (Tensor): Ground truth lines
                [num_gt, num_points, 2]
        Returns:
            torch.Tensor: reg_cost value with weight
                shape [num_pred, num_gt]
        """

        num_gts, num_bboxes = gt_lines.size(0), lines_pred.size(0)

        dist_mat = lines_pred.new_full((num_bboxes, num_gts),
                                       1.0,)

        for i in range(num_bboxes):
            for j in range(num_gts):
                dist_mat[i, j] = chamfer_distance(
                    lines_pred[i], gt_lines[j])

        return dist_mat * self.weight


@MATCH_COST.register_module()
class LinesCost(object):
    """LinesL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, lines_pred, gt_lines, **kwargs):
        """
        Args:
            lines_pred (Tensor): predicted normalized lines:
                [num_query, num_points, 2]
            gt_lines (Tensor): Ground truth lines
                [num_gt, num_points, 2]
        Returns:
            torch.Tensor: reg_cost value with weight
                shape [num_pred, num_gt]
        """
        gt_revser = torch.flip(gt_lines, dims=[-2])
        gt_revser_flat = gt_revser.flatten(1, 2)

        pred_flat = lines_pred.flatten(1, 2)
        gt_flat = gt_lines.flatten(1, 2)

        div_ = pred_flat.size(-1)

        dist_mat = torch.cdist(pred_flat, gt_flat, p=1) / div_

        return dist_mat * self.weight


@MATCH_COST.register_module()
class BBoxCostC:
    """BBoxL1Cost.
     Args:
         weight (int | float, optional): loss_weight
         box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN
     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import BBoxL1Cost
         >>> import torch
         >>> self = BBoxL1Cost()
         >>> bbox_pred = torch.rand(1, 4)
         >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(bbox_pred, gt_bboxes, factor)
         tensor([[1.6172, 1.6422]])
    """

    def __init__(self, weight=1., box_format='xyxy'):
        self.weight = weight
        assert box_format in ['xyxy', 'xywh']
        self.box_format = box_format

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        # if self.box_format == 'xywh':
        #     gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
        # elif self.box_format == 'xyxy':
        #     bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@MATCH_COST.register_module()
class IoUCostC:
    """IoUCost.
     Args:
         iou_mode (str, optional): iou mode such as 'iou' | 'giou'
         weight (int | float, optional): loss weight
     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import IoUCost
         >>> import torch
         >>> self = IoUCost()
         >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
         >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> self(bboxes, gt_bboxes)
         tensor([[-0.1250,  0.1667],
                [ 0.1667, -0.5000]])
    """

    def __init__(self, iou_mode='giou', weight=1., box_format='xywh'):
        self.weight = weight
        self.iou_mode = iou_mode
        assert box_format in ['xyxy', 'xywh']
        self.box_format = box_format

    def __call__(self, bboxes, gt_bboxes):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            torch.Tensor: iou_cost value with weight
        """
        if self.box_format == 'xywh':
            bboxes = bbox_cxcywh_to_xyxy(bboxes)
            gt_bboxes = bbox_cxcywh_to_xyxy(gt_bboxes)

        # overlaps: [num_bboxes, num_gt]
        overlaps = bbox_overlaps(
            bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)
        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * self.weight

@MATCH_COST.register_module()
class DynamicLinesCost(object):
    """LinesL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, lines_pred, lines_gt, masks_pred, masks_gt):
        """
        Args:
            lines_pred (Tensor): predicted normalized lines:
                [nP, num_points, 2]
            lines_gt (Tensor): Ground truth lines
                [nG, num_points, 2]
            masks_pred: [nP, num_points]
            masks_gt: [nG, num_points]
        Returns:
            dist_mat: reg_cost value with weight
                shape [nP, nG]
        """

        dist_mat = self.cal_dist(lines_pred, lines_gt)

        dist_mat = self.get_dynamic_line(dist_mat, masks_pred, masks_gt)

        dist_mat = dist_mat * self.weight

        return dist_mat

    def cal_dist(self, x1, x2):
        '''
            Args:
                x1: B1,N,2
                x2: B2,N,2
            Return:
                dist_mat: B1,B2,N
        '''
        x1 = x1.permute(1, 0, 2)
        x2 = x2.permute(1, 0, 2)

        dist_mat = torch.cdist(x1, x2, p=2)

        dist_mat = dist_mat.permute(1, 2, 0)

        return dist_mat

    def get_dynamic_line(self, mat, m1, m2):
        '''
            get dynamic line with difference approach
            mat: N1xN2xnpts
            m1: N1xnpts
            m2: N2xnpts
        '''

        # nPxnGxnum_points
        m1 = m1.unsqueeze(1).sigmoid() > 0.5
        m2 = m2.unsqueeze(0)

        valid_points_mask = (m1 + m2)/2.

        average_factor_mask = valid_points_mask.sum(-1) > 0
        average_factor = average_factor_mask.masked_fill(
            ~average_factor_mask, 1)

        # takes the average
        mat = mat * valid_points_mask
        mat = mat.sum(-1) / average_factor

        return mat


@MATCH_COST.register_module()
class BBoxLogitsCost(object):
    """BBoxLogits.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def calNLL(self, logits, value):
        '''
            Args:
                logits: B1, 8, cls_dim
                value: B2, 8,
            Return:
                log_likelihood: B1,B2,8
        '''

        logits = logits[:, None]
        value = value[None]

        value = value.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, logits)
        value = value[..., :1]
        return log_pmf.gather(-1, value).squeeze(-1)

    def __call__(self, bbox_pred, bbox_gt, **kwargs):
        """
        Args:
            bbox_pred: nproposal, 4*2, pos_dim
            bbox_gt: ngt, 4*2
        Returns:
            cost: nproposal, ngt
        """

        cost = self.calNLL(bbox_pred, bbox_gt).mean(-1)

        return cost * self.weight


@MATCH_COST.register_module()
class MapQueriesCost(object):

    def __init__(self, cls_cost, reg_cost, iou_cost=None):

        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)

        self.iou_cost = None
        if iou_cost is not None:
            self.iou_cost = build_match_cost(iou_cost)

    def __call__(self, preds: dict, gts: dict):

        # classification and bboxcost.
        cls_cost = self.cls_cost(preds['scores'], gts['labels'])

        # regression cost
        regkwargs = {}
        if 'masks' in preds and 'masks' in gts:
            assert isinstance(self.reg_cost, DynamicLinesCost), ' Issues!!'
            regkwargs = {
                'masks_pred': preds['masks'],
                'masks_gt': gts['masks'],
            }

        reg_cost = self.reg_cost(preds['lines'], gts['lines'], **regkwargs)

        # weighted sum of above three costs
        cost = cls_cost + reg_cost

        # Iou
        if self.iou_cost is not None:
            iou_cost = self.iou_cost(preds['lines'],gts['lines'])
            cost += iou_cost


        return cost
