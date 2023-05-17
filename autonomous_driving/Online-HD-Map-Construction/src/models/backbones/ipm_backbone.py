import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.builder import BACKBONES
from mmdet.models import build_backbone, build_neck

class UpsampleBlock(nn.Module):
    def __init__(self, ins, outs):
        super(UpsampleBlock, self).__init__()
        self.gn = nn.GroupNorm(32, outs)
        self.conv = nn.Conv2d(ins, outs, kernel_size=3,
                              stride=1, padding=1)  # same
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.relu(self.gn(x))
        x = self.upsample2x(x)

        return x

    def upsample2x(self, x):
        _, _, h, w = x.shape
        x = F.interpolate(x, size=(h*2, w*2),
                          mode='bilinear', align_corners=True)
        return x


class Upsample(nn.Module):

    def __init__(self,
                 zoom_size=(2, 4, 8),
                 in_channels=128,
                 out_channels=128,
                 ):
        super(Upsample, self).__init__()

        self.out_channels = out_channels

        input_conv = UpsampleBlock(in_channels, out_channels)
        inter_conv = UpsampleBlock(out_channels, out_channels)

        fscale = []
        for scale_factor in zoom_size:

            layer_num = int(math.log2(scale_factor))
            if layer_num < 1:
                fscale.append(nn.Identity())
                continue

            tmp = [copy.deepcopy(input_conv), ]
            tmp += [copy.deepcopy(inter_conv) for i in range(layer_num-1)]
            fscale.append(nn.Sequential(*tmp))

        self.fscale = nn.ModuleList(fscale)

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, imgs):

        rescale_i = []
        for f, img in zip(self.fscale, imgs):
            rescale_i.append(f(img))

        out = sum(rescale_i)

        return out


@BACKBONES.register_module()
class IPMEncoder(nn.Module):
    '''
    encode cam features
    '''

    def __init__(self,
                 img_backbone,
                 img_neck,
                 upsample,
                 xbound=[-30.0, 30.0, 0.5],
                 ybound=[-15.0, 15.0, 0.5],
                 zbound=[-10.0, 10.0, 20.0],
                 heights=[-1.1, 0, 0.5, 1.1],
                 pretrained=None,
                 out_channels=128,
                 num_cam=6,
                 use_lidar=False,
                 use_image=True,
                 lidar_dim=128,
                 ):
        super(IPMEncoder, self).__init__()
        self.x_bound = xbound
        self.y_bound = ybound
        self.heights = heights

        self.num_cam = num_cam

        num_x = int((xbound[1] - xbound[0]) / xbound[2])
        num_y = int((ybound[1] - ybound[0]) / ybound[2])

        self.img_backbone = build_backbone(img_backbone)
        self.img_neck = build_neck(img_neck)
        self.upsample = Upsample(**upsample)

        self.use_image = use_image
        self.use_lidar = use_lidar
        if self.use_lidar:
            self.pp = PointPillarEncoder(lidar_dim, xbound, ybound, zbound)

            self.outconvs =\
                nn.Conv2d((self.upsample.out_channels+3)*len(heights), out_channels//2, 
                            kernel_size=3, stride=1, padding=1)  # same
            if self.use_image:
                _out_channels = out_channels//2
            else:
                _out_channels = out_channels

            self.outconvs_lidar =\
                nn.Conv2d(lidar_dim, _out_channels, 
                            kernel_size=3, stride=1, padding=1)  # same
        else:
            self.outconvs =\
                nn.Conv2d((self.upsample.out_channels+3)*len(heights), out_channels, 
                            kernel_size=3, stride=1, padding=1)  # same

        self.init_weights(pretrained=pretrained)

        # bev_plane
        bev_planes = [construct_plane_grid(
            xbound, ybound, h) for h in self.heights]
        self.register_buffer('bev_planes', torch.stack(
            bev_planes),)  # nlvl,bH,bW,2

        self.masked_embeds = nn.Embedding(len(heights), out_channels)


    def init_weights(self, pretrained=None):
        """Initialize model weights."""

        self.img_backbone.init_weights()
        self.img_neck.init_weights()
        self.upsample.init_weights()

        for p in self.outconvs.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        if self.use_lidar:
            for p in self.outconvs_lidar.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            
            for p in self.pp.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def extract_img_feat(self, imgs):
        '''
            Extract image feaftures and sum up into one pic
            Args:
                imgs: B, n_cam, C, iH, iW
            Returns: 
                img_feat: B * n_cam, C, H, W
        '''

        B, n_cam, C, iH, iW = imgs.shape
        imgs = imgs.view(B * n_cam, C, iH, iW)

        img_feats = self.img_backbone(imgs)

        # reduce the channel dim
        img_feats = self.img_neck(img_feats)

        # fuse four feature map
        img_feat = self.upsample(img_feats)

        return img_feat

    def forward(self, imgs, img_metas, *args, points=None, **kwargs):
        '''
            Args: 
                imgs: torch.Tensor of shape [B, N, 3, H, W]
                    N: number of cams
                img_metas: 
                    # N=6, ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
                    ego2cam: [B, N, 4, 4] 
                    cam_intrinsics: [B, N, 3, 3]
                    cam2ego_rotations: [B, N, 3, 3]
                    cam2ego_translations: [B, N, 3]
                    ...
            Outs:
                bev_feature: torch.Tensor of shape [B, C*nlvl, bH, bW]
        '''

        if self.use_image:
            self.B = imgs.shape[0]

            # Get transform matrix
            ego2cam = []
            for img_meta in img_metas:
                ego2cam.append(img_meta['ego2img'])
            img_shape = imgs.shape[-2:]

            ego2cam = np.asarray(ego2cam)
            # Image backbone
            img_feats = self.extract_img_feat(imgs)

            # IPM
            bev_feat, bev_feat_mask = self.ipm(img_feats, ego2cam, img_shape)

            # multi level into a same
            bev_feat = bev_feat.flatten(1, 2)
            bev_feat = self.outconvs(bev_feat)

        if self.use_lidar:
            lidar_feat = self.get_lidar_feature(points)
            if self.use_image:
                bev_feat = torch.cat([bev_feat,lidar_feat],dim=1)
            else:
                bev_feat = lidar_feat

        return bev_feat

    def ipm(self, cam_feat, ego2cam, img_shape):
        '''
            inverse project 
            Args:
                cam_feat: B*ncam, C, cH, cW
                img_shape: tuple(H, W)
            Returns:
                project_feat: B, C, nlvl, bH, bW
                bev_feat_mask: B, 1, nlvl, bH, bW
        '''
        C = cam_feat.shape[1]
        bev_grid = self.bev_planes.unsqueeze(0).repeat(self.B, 1, 1, 1, 1)
        nlvl, bH, bW = bev_grid.shape[1:4]
        bev_grid = bev_grid.flatten(1, 3)  # B, nlvl*W*H, 3

        # Find points in cam coords
        # bev_grid_pos: B*ncam, nlvl*bH*bW, 2
        bev_grid_pos, bev_cam_mask = get_campos(bev_grid, ego2cam, img_shape)
        # B*cam, nlvl*bH, bW, 2
        bev_grid_pos = bev_grid_pos.unflatten(-2, (nlvl*bH, bW))

        # project feat from 2D to bev plane
        projected_feature = F.grid_sample(
            cam_feat, bev_grid_pos, align_corners=False).view(self.B, -1, C, nlvl, bH, bW)  # B,cam,C,nlvl,bH,bW

        # B,cam,nlvl,bH,bW
        bev_feat_mask = bev_cam_mask.unflatten(-1, (nlvl, bH, bW))

        # eliminate the ncam
        # The bev feature is the sum of the 6 cameras
        bev_feat_mask = bev_feat_mask.unsqueeze(2)
        projected_feature = (projected_feature*bev_feat_mask).sum(1)
        num_feat = bev_feat_mask.sum(1)

        projected_feature = projected_feature / \
            num_feat.masked_fill(num_feat == 0, 1)

        # concatenate a position information
        # projected_feature: B, bH, bW, nlvl, C+3
        bev_grid = bev_grid.view(self.B, nlvl, bH, bW,
                                 3).permute(0, 4, 1, 2, 3)
        projected_feature = torch.cat(
            (projected_feature, bev_grid), dim=1)

        return projected_feature, bev_feat_mask.sum(1) > 0

    def get_lidar_feature(self, points):
        ptensor, pmask = points
        lidar_feature = self.pp(ptensor, pmask)

        # bev_grid = self.bev_planes[...,:-1].unsqueeze(0).repeat(self.B, 1, 1, 1, 1)
        # bev_grid = bev_grid[:,0]

        # bev_grid = bev_grid.permute(0, 3, 1, 2)
        # lidar_feature = torch.cat(
        #     (lidar_feature, bev_grid), dim=1)
        
        lidar_feature = self.outconvs_lidar(lidar_feature)

        return lidar_feature


def construct_plane_grid(xbound, ybound, height: float, dtype=torch.float32):
    '''
        Returns:
            plane: H, W, 3
    '''

    xmin, xmax = xbound[0], xbound[1]
    num_x = int((xbound[1] - xbound[0]) / xbound[2])
    ymin, ymax = ybound[0], ybound[1]
    num_y = int((ybound[1] - ybound[0]) / ybound[2])

    x = torch.linspace(xmin, xmax, num_x, dtype=dtype)
    y = torch.linspace(ymin, ymax, num_y, dtype=dtype)

    # [num_y, num_x]
    y, x = torch.meshgrid(y, x)

    z = torch.ones_like(x) * height

    # [num_y, num_x, 3]
    plane = torch.stack([x, y, z], dim=-1)

    return plane


def get_campos(reference_points, ego2cam, img_shape):
    '''
        Find the each refence point's corresponding pixel in each camera
        Args: 
            reference_points: [B, num_query, 3]
            ego2cam: (B, num_cam, 4, 4)
        Outs:
            reference_points_cam: (B*num_cam, num_query, 2)
            mask:  (B, num_cam, num_query)
            num_query == W*H
    '''

    ego2cam = reference_points.new_tensor(ego2cam)  # (B, N, 4, 4)
    reference_points = reference_points.clone()

    B, num_query = reference_points.shape[:2]
    num_cam = ego2cam.shape[1]

    # reference_points (B, num_queries, 4)
    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1)
    reference_points = reference_points.view(
        B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)

    ego2cam = ego2cam.view(
        B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)

    # reference_points_cam (B, num_cam, num_queries, 4)
    reference_points_cam = (ego2cam @ reference_points).squeeze(-1)

    eps = 1e-9
    mask = (reference_points_cam[..., 2:3] > eps)

    reference_points_cam =\
        reference_points_cam[..., 0:2] / \
        reference_points_cam[..., 2:3] + eps

    reference_points_cam[..., 0] /= img_shape[1]
    reference_points_cam[..., 1] /= img_shape[0]

    # from 0~1 to -1~1
    reference_points_cam = (reference_points_cam - 0.5) * 2

    mask = (mask & (reference_points_cam[..., 0:1] > -1.0)
                 & (reference_points_cam[..., 0:1] < 1.0)
                 & (reference_points_cam[..., 1:2] > -1.0)
                 & (reference_points_cam[..., 1:2] < 1.0))

    # (B, num_cam, num_query)
    mask = mask.view(B, num_cam, num_query)
    reference_points_cam = reference_points_cam.view(B*num_cam, num_query, 2)

    return reference_points_cam, mask


def _test():
    pass


if __name__ == '__main__':
    _test()
