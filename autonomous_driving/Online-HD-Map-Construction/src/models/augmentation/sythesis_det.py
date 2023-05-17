import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseSythesis(nn.Module):

    def __init__(self, 
            p, scale=0.01, shift_scale=(8,5), 
            scaling_size=(0.1,0.1), canvas_size=(200, 100),
            bbox_type='sce',
            poly_coord_dim=2,
            bbox_coord_dim=2,
            quantify=True):
        super(NoiseSythesis, self).__init__()

        self.p = p
        self.scale = scale
        self.bbox_type = bbox_type
        self.quantify = quantify

        self.poly_coord_dim = poly_coord_dim
        self.bbox_coord_dim = bbox_coord_dim

        self.transforms = [self.random_shifting, self.random_scaling]
        # self.transforms = [self.random_scaling]

        self.register_buffer('canvas_size', torch.tensor(canvas_size))
        self.register_buffer('shift_scale', torch.tensor(shift_scale).float())
        self.register_buffer('scaling_size', torch.tensor(scaling_size))

    def random_scaling(self, bbox):
        '''
            bbox: B, paramter_num, 2
        '''
        device = bbox.device
        dtype = bbox.dtype
        B = bbox.shape[0]

        noise = (torch.rand(B, device=device)*2-1)[:,None,None] # [-1,1]
        scale = self.scaling_size.to(device)
        scale = (noise * scale) + 1

        scaled_bbox = bbox * scale

        # recenterization
        coffset = scaled_bbox.mean(-2) - bbox.float().mean(-2)
        scaled_bbox = scaled_bbox - coffset[:,None]

        return scaled_bbox.round().type(dtype)

    def random_shifting(self, bbox):
        '''
            bbox: B, paramter_num, 2
        '''
        device = bbox.device
        batch_size = bbox.shape[0]

        shift_scale = self.shift_scale
        scale = (bbox.max(1)[0] - bbox.min(1)[0]) * 0.1
        scale = torch.where(scale < shift_scale, scale, shift_scale)

        noise = (torch.rand(batch_size, 2, device=device)*2-1) # [-1,1]
        offset = (noise * scale).round().type(bbox.dtype)

        shifted_bbox = bbox + offset[:, None]
        
        return shifted_bbox
    
    def gaussian_noise_bbox(self, bbox):

        dtype = bbox.dtype
        batch_size = bbox.shape[0]

        scale = (self.canvas_size * self.scale)[:self.bbox_coord_dim]

        noisy_bbox = torch.normal(bbox.type(torch.float), scale)

        if self.quantify:
            noisy_bbox = noisy_bbox.round().type(dtype)
            # prevent out of bound case
            for i in range(self.bbox_coord_dim):
                noisy_bbox[...,i] =\
                    torch.clamp(noisy_bbox[...,0],1,self.canvas_size[i])
        else:
            noisy_bbox = noisy_bbox.type(torch.float)
        
        return noisy_bbox
    
    def gaussian_noise_poly(self, polyline, polyline_mask):

        device = polyline.device
        batchsize = polyline.shape[0]
        scale = self.canvas_size * self.scale

        polyline = F.pad(polyline,(0,self.poly_coord_dim-1))
        polyline = polyline.view(batchsize,-1, self.poly_coord_dim)
        mask = F.pad(polyline_mask[:,1:],(0,self.poly_coord_dim))
        
        noisy_polyline = torch.normal(polyline.type(torch.float), scale)

        if self.quantify:
            noisy_polyline = noisy_polyline.round().type(polyline.dtype)

            # prevent out of bound case
            for i in range(self.poly_coord_dim):
                noisy_polyline[...,i] =\
                    torch.clamp(noisy_polyline[...,i],0,self.canvas_size[i])

        else:
            noisy_polyline = noisy_polyline.type(torch.float)

        noisy_polyline = noisy_polyline.view(batchsize,-1) * mask
        noisy_polyline = noisy_polyline[:,:-(self.poly_coord_dim-1)]

        return noisy_polyline

    def random_apply(self, bbox):

        for t in self.transforms:

            if self.p < torch.rand(1):
                continue

            bbox = t(bbox)

        # prevent out of bound case
        bbox[...,0] =\
            torch.clamp(bbox[...,0],0,self.canvas_size[0])
        
        bbox[...,1] =\
            torch.clamp(bbox[...,1],0,self.canvas_size[1])

        return bbox

    def simple_aug(self, batch):

        # augment bbox
        if self.bbox_type in ['sce', 'xyxy']:
            fbbox = batch['bbox_flat']
            seq_len = fbbox.shape[0]
            bbox = fbbox.view(seq_len, -1, 2)
            bbox = self.gaussian_noise_bbox(bbox)
            fbbox_aug = bbox.view(seq_len, -1)

            aug_mask = torch.rand(fbbox.shape,device=fbbox.device)
            fbbox = torch.where(aug_mask<self.p, fbbox_aug, fbbox)
        elif self.bbox_type == 'rxyxy':
            fbbox = self.rbbox_aug(batch)
        elif self.bbox_type == 'convex_hull':
            fbbox = self.convex_hull_aug(batch)

        # augment
        polyline = batch['polylines']
        polyline_mask = batch['polyline_masks']
        polyline_aug = self.gaussian_noise_poly(polyline, polyline_mask)
        
        aug_mask = torch.rand(polyline.shape,device=polyline.device)
        polyline = torch.where(aug_mask<self.p, polyline_aug, polyline)

        return polyline, fbbox

    def rbbox_aug(self, batch):
        
        return None
    
    def convex_hull_aug(self,batch):
    
        return None

    def __call__(self, batch, simple_aug=False):

        if simple_aug:

            return self.simple_aug(batch)

        else:
            fbbox = batch['bbox_flat']
            seq_len = fbbox.shape[0]
            bbox = fbbox.view(seq_len, -1, self.bbox_coord_dim)

            aug_bbox = self.random_apply(bbox)

            aug_bbox_flat = aug_bbox.view(seq_len, -1)


        return aug_bbox_flat
