import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from mmdet.models import HEADS
from .detgen_utils.causal_trans import (CausalTransformerDecoder,
                           CausalTransformerDecoderLayer)
from .detgen_utils.utils import (dequantize_verts, generate_square_subsequent_mask,
                    quantize_verts, top_k_logits, top_p_logits)
from mmcv.runner import force_fp32, auto_fp16

@HEADS.register_module(force=True)
class PolylineGenerator(nn.Module):
    """
      Autoregressive generative model of n-gon meshes.
      Operates on sets of input vertices as well as flattened face sequences with
      new face and stopping tokens:
      [f_0^0, f_0^1, f_0^2, NEW, f_1^0, f_1^1, ..., STOP]
      Input vertices are encoded using a Transformer encoder.
      Input face sequences are embedded and tagged with learned position indicators,
      as well as their corresponding vertex embeddings. A transformer decoder
      outputs a pointer which is compared to each vertex embedding to obtain a
      distribution over vertex indices.
    """

    def __init__(self,
                 in_channels,
                 encoder_config,
                 decoder_config,
                 class_conditional=True,
                 num_classes=55,
                 decoder_cross_attention=True,
                 use_discrete_vertex_embeddings=True,
                 condition_points_num=3,
                 coord_dim=2,
                 canvas_size=(400, 200),
                 max_seq_length=500,
                 name='gen_model'):
        """Initializes FaceModel.
          Args:
            encoder_config: Dictionary with TransformerEncoder config.
            decoder_config: Dictionary with TransformerDecoder config.
            class_conditional: If True, then condition on learned class embeddings.
            num_classes: Number of classes to condition on.
            decoder_cross_attention: If True, the use cross attention from decoder
              querys into encoder outputs.
            use_discrete_vertex_embeddings: If True, use discrete vertex embeddings.
            max_seq_length: Maximum face sequence length. Used for learned position
              embeddings.
            name: Name of variable scope
        """
        super(PolylineGenerator, self).__init__()
        self.embedding_dim = decoder_config['layer_config']['d_model']
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length
        self.decoder_cross_attention = decoder_cross_attention
        self.use_discrete_vertex_embeddings = use_discrete_vertex_embeddings

        self.condition_points_num = condition_points_num
        self.fp16_enabled = False

        self.coord_dim = coord_dim  # if we use xyz else 2 when we use xy
        self.kp_coord_dim = coord_dim if coord_dim==2 else 2 # XXX
        self.register_buffer('canvas_size', torch.tensor(canvas_size))

        # initialize the model
        self._project_to_logits = nn.Linear(
            self.embedding_dim,
            max(canvas_size) + 1,  # + 1 for stopping token. use_bias=True,
        )

        self.input_proj = nn.Conv2d(
            in_channels, self.embedding_dim, kernel_size=1)

        decoder_layer = CausalTransformerDecoderLayer(
            **decoder_config.pop('layer_config'))
        self.decoder = CausalTransformerDecoder(
            decoder_layer, **decoder_config)

        self._init_embedding()
        self.init_weights()

    def _init_embedding(self):

        if self.class_conditional:
            self.label_embed = nn.Embedding(
                self.num_classes, self.embedding_dim)

        self.coord_embed = nn.Embedding(self.coord_dim, self.embedding_dim)
        self.pos_embeddings = nn.Embedding(
            self.max_seq_length, self.embedding_dim)

        # to indicate the role of the position is the start of the line or the end of it.
        self.bbox_context_embed = \
            nn.Embedding(self.condition_points_num, self.embedding_dim)

        self.img_coord_embed = nn.Linear(2, self.embedding_dim)

        # initialize the verteices embedding
        if self.use_discrete_vertex_embeddings:
            self.vertex_embed = nn.Embedding(
                max(self.canvas_size) + 1, self.embedding_dim)
        else:
            self.vertex_embed = nn.Linear(1, self.embedding_dim)

    def init_weights(self):

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _embed_kps(self, bbox):

        bbox_len = bbox.shape[-1]
        # Bbox_context
        bbox_embedding = self.bbox_context_embed(
            (torch.arange(bbox_len, device=bbox.device) / self.kp_coord_dim).floor().long())

        # Coord indicators (x, y)
        coord_embeddings = self.coord_embed(
            torch.arange(bbox_len, device=bbox.device) % self.kp_coord_dim)

        # Discrete vertex value embeddings
        vert_embeddings = self.vertex_embed(bbox)

        return vert_embeddings + (bbox_embedding+coord_embeddings)[None]

    def _prepare_context(self, batch, context):
        """Prepare class label and vertex context."""

        global_context_embedding = None
        if self.class_conditional:
            global_context_embedding = self.label_embed(batch['lines_cls'])

        bbox_embeddings = self._embed_kps(batch['bbox_flat'])

        if global_context_embedding is not None:
            global_context_embedding = torch.cat(
                [global_context_embedding[:, None], bbox_embeddings], dim=1)

        # Pass images through encoder
        image_embeddings = assign_bev(
            context['bev_embeddings'], batch['lines_bs_idx'])
        image_embeddings = self.input_proj(image_embeddings)

        device = image_embeddings.device

        # Add 2D coordinate grid embedding
        H, W = image_embeddings.shape[2:]
        Ws = torch.linspace(-1., 1., W)
        Hs = torch.linspace(-1., 1., H)
        image_coords = torch.stack(
            torch.meshgrid(Hs, Ws), dim=-1).to(device)
        image_coord_embeddings = self.img_coord_embed(image_coords)

        image_embeddings += image_coord_embeddings[None].permute(0, 3, 1, 2)

        # Reshape spatial grid to sequence
        B = image_embeddings.shape[0]
        sequential_context_embeddings = image_embeddings.reshape(
            B, self.embedding_dim, -1).permute(0, 2, 1)

        return (global_context_embedding,
                sequential_context_embeddings)

    def _embed_inputs(self, seqs, condition_embedding=None):
        """Embeds face sequences and adds within and between face positions.
          Args:
            seq: B, seqlen=vlen*3, 
            condition_embedding: B, [c,xs,ys,xe,ye](5), h
          Returns:
            embeddings: B, seqlen, h
        """
        B, seq_len = seqs.shape[:2]

        # Position embeddings
        pos_embeddings = self.pos_embeddings(
            (torch.arange(seq_len, device=seqs.device) / self.coord_dim).floor().long())  # seq_len, h

        # Coord indicators (x, y, z(optional))
        coord_embeddings = self.coord_embed(
            torch.arange(seq_len, device=seqs.device) % self.coord_dim)

        # Discrete vertex value embeddings
        vert_embeddings = self.vertex_embed(seqs)

        # Aggregate embeddings
        embeddings = vert_embeddings + \
            (coord_embeddings+pos_embeddings)[None]
        embeddings = torch.cat([condition_embedding, embeddings], dim=1)

        return embeddings

    def forward(self, batch: dict, **kwargs):
        """
          Pass batch through face model and get log probabilities.
          Args:
            batch: Dictionary containing:
              'vertices_dequantized': Tensor of shape [batch_size, num_vertices, 3].
              'faces': int32 tensor of shape [batch_size, seq_length] with flattened
                faces.
              'vertices_mask': float32 tensor with shape
                [batch_size, num_vertices] that masks padded elements in 'vertices'.
        """

        if self.training:
            return self.forward_train(batch, **kwargs)
        else:
            return self.inference(batch, **kwargs)
        
    def sperate_forward(self, batch, context, **kwargs):

        polyline_length = batch['polyline_masks'].sum(-1)
        c1, c2, revert_idx, size = get_chunk_idx(polyline_length)

        sizes = [size, polyline_length.max()]
        polyline_logits = []
        for c_idx, size in zip([c1,c2], sizes):

            new_batch = assign_batch(batch,c_idx, size)
            _poly_logits = self._forward_train(new_batch,context,**kwargs)
            polyline_logits.append(_poly_logits)
        
        # maybe imporve the speed 
        for i, (_poly_logits, size) in enumerate(zip(polyline_logits, sizes)):    
            if size < sizes[1]:
                _poly_logits = F.pad(_poly_logits, (0,0,0,sizes[1]-size), "constant", 0)
                polyline_logits[i] = _poly_logits
        
        polyline_logits = torch.cat(polyline_logits,0)
        polyline_logits = polyline_logits[revert_idx]
        cat_dist = Categorical(logits=polyline_logits)

        return {'polylines':cat_dist}    
            

    def forward_train(self, batch: dict, context: dict, **kwargs):
        """
          Returns:
            pred_dist: Categorical predictive distribution with batch shape
                [batch_size, seq_length].
        """
        # we use the gt vertices
        if False:
            polyline_logits = self._forward_train(batch, context, **kwargs)
            cat_dist = Categorical(logits=polyline_logits)
            return {'polylines':cat_dist}
        else:
            return self.sperate_forward(batch, context, **kwargs)

    def _forward_train(self, batch: dict, context: dict, **kwargs):
        """
          Returns:
            pred_dist: Categorical predictive distribution with batch shape
                [batch_size, seq_length].
        """
        # we use the gt vertices
        global_context, seq_context = self._prepare_context(
            batch, context)
                
        logits = self.body(
            # Last element not used for preds
            batch['polylines'][:, :-1],
            global_context_embedding=global_context,
            sequential_context_embeddings=seq_context,
            return_logits=True,
            is_training=self.training)

        return logits

    @force_fp32(apply_to=('global_context_embedding','sequential_context_embeddings','cache'))
    def body(self,
             seqs,
             global_context_embedding=None,
             sequential_context_embeddings=None,
             temperature=1.,
             top_k=0,
             top_p=1.,
             cache=None,
             return_logits=False,
             is_training=True):
        """
            Outputs categorical dist for vertex indices.
            Body of the face model
        """

        # Embed inputs
        condition_len = global_context_embedding.shape[1]
        decoder_inputs = self._embed_inputs(
            seqs, global_context_embedding)

        # Pass through Transformer decoder
        # since our memory efficient decoder only support seq first setting.
        decoder_inputs = decoder_inputs.transpose(0, 1)
        if sequential_context_embeddings is not None:
            sequential_context_embeddings = sequential_context_embeddings.transpose(
                0, 1)

        causal_msk = None
        if is_training:
            causal_msk = generate_square_subsequent_mask(
                decoder_inputs.shape[0], condition_len=condition_len, device=decoder_inputs.device)
        
        decoder_outputs, cache = self.decoder(
            tgt=decoder_inputs,
            cache=cache,
            memory=sequential_context_embeddings,
            causal_mask=causal_msk,
        )

        decoder_outputs = decoder_outputs.transpose(0, 1)

        # since we only need the predict seq
        decoder_outputs = decoder_outputs[:, condition_len-1:]

        # Get logits and optionally process for sampling
        logits = self._project_to_logits(decoder_outputs)

        # y mask
        _vert_mask = torch.arange(logits.shape[-1], device=logits.device)
        vertices_mask_y = (_vert_mask < self.canvas_size[1]+1)
        vertices_mask_y[0] = False # y position doesn't have stop sign 
        logits[:, 1::self.coord_dim] = logits[:, 1::self.coord_dim] * \
            vertices_mask_y - ~vertices_mask_y*1e9

        if self.coord_dim > 2:
            # z mask
            _vert_mask = torch.arange(logits.shape[-1], device=logits.device)
            vertices_mask_z = (_vert_mask < self.canvas_size[2]+1)
            vertices_mask_z[0] = False # y position doesn't have stop sign 
            logits[:, 2::self.coord_dim] = logits[:, 2::self.coord_dim] * \
                vertices_mask_z - ~vertices_mask_z*1e9
        

        logits = logits/temperature
        logits = top_k_logits(logits, top_k)
        logits = top_p_logits(logits, top_p)
        if return_logits:
            return logits

        cat_dist = Categorical(logits=logits)

        return cat_dist, cache

    @force_fp32(apply_to=('pred'))
    def loss(self, gt: dict, pred: dict):

        weight = gt['polyline_weights']
        mask = gt['polyline_masks']
        
        loss = -torch.sum(
            pred['polylines'].log_prob(gt['polylines']) * mask * weight)/weight.sum()

        return {'seq': loss}

    def inference(self,
                  batch: dict,
                  context: dict,
                  max_sample_length=None,
                  temperature=1.,
                  top_k=0,
                  top_p=1.,
                  only_return_complete=False,
                  gt_condition=False,
                  **kwargs):
        """Sample from face model using caching.
        Args:
            context: Dictionary of context, including 'vertices' and 'vertices_mask'.
                See _prepare_context for details.
            max_sample_length: Maximum length of sampled vertex sequences. Sequences
                that do not complete are truncated.
            temperature: Scalar softmax temperature > 0.
            top_k: Number of tokens to keep for top-k sampling.
            top_p: Proportion of probability mass to keep for top-p sampling.
            only_return_complete: If True, only return completed samples. Otherwise
                return all samples along with completed indicator.
        Returns:
            outputs: Output dictionary with fields:
                'completed': Boolean tensor of shape [num_samples]. If True then
                corresponding sample completed within max_sample_length.
                'faces': Tensor of samples with shape [num_samples, num_verts, 3].
                'valid_polyline_len': Tensor indicating number of vertices for each
                example in padded vertex samples.
        """
        # prepare the conditional variable
        global_context, seq_context = self._prepare_context(
            batch, context)
        device = global_context.device
        batch_size = global_context.shape[0]

        # While loop sampling with caching
        samples = torch.empty(
            [batch_size, 0], dtype=torch.int32, device=device)
        max_sample_length = max_sample_length or self.max_seq_length
        seq_len = max_sample_length*self.coord_dim+1
        cache = None

        decoded_tokens = \
            torch.zeros((batch_size,seq_len),
                device=device,dtype=torch.long)
        remain_idx = torch.arange(batch_size, device=device)
        for i in range(seq_len):


            # While-loop body for autoregression calculation.
            pred_dist, cache = self.body(
                samples,
                global_context_embedding=global_context,
                sequential_context_embeddings=seq_context,
                cache=cache,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                is_training=False)
            samples = pred_dist.sample()

            decoded_tokens[remain_idx,i] = samples[:,-1]

            # Stopping conditions for autoregressive calculation.
            if not (decoded_tokens[:,:i+1] != 0).all(-1).any():
                break
            
            # update state, check the new position is zero.
            valid_idx = (samples[:,-1] != 0).nonzero(as_tuple=True)[0]
            remain_idx = remain_idx[valid_idx]
            cache = cache[:,:,valid_idx]
            global_context = global_context[valid_idx]
            seq_context = seq_context[valid_idx]
            samples = samples[valid_idx]

        # decoded_tokens = torch.cat(decoded_tokens,dim=1)
        decoded_tokens = decoded_tokens[:,:i+1]

        outputs = self.post_process(decoded_tokens, seq_len,
                                    device, only_return_complete)

        return outputs

    def post_process(self, polyline,
                     max_seq_len=None,
                     device=None,
                     only_return_complete=True):
        '''
            format the predictions
            find the mask
        '''
        # Record completed samples
        complete_samples = (polyline == 0).any(-1)

        # Find number of faces
        sample_seq_length = polyline.shape[-1]
        _polyline_mask = torch.arange(sample_seq_length)[None].to(device)

        # Get largest stopping point for incomplete samples.
        valid_polyline_len = torch.full_like(polyline[:,0], sample_seq_length)
        zero_inds = (polyline == 0).type(torch.int32).argmax(-1)

        # Real length
        valid_polyline_len[complete_samples] = zero_inds[complete_samples] + 1
        polyline_mask = _polyline_mask < valid_polyline_len[:, None]

        # Mask faces beyond stopping token with zeros
        polyline = polyline*polyline_mask

        # Pad to maximum size with zeros
        pad_size = max_seq_len - sample_seq_length
        polyline = F.pad(polyline, (0, pad_size))
        # polyline_mask = F.pad(polyline_mask, (0, pad_size))

        # XXX
        # if only_return_complete:
        #     polyline = polyline[complete_samples]
        #     valid_polyline_len = valid_polyline_len[complete_samples]
        #     context = tf.nest.map_structure(
        #         lambda x: tf.boolean_mask(x, complete_samples), context)
        #     complete_samples = complete_samples[complete_samples]

        # outputs
        outputs = {
            'completed': complete_samples,
            'polylines': polyline,
            'polyline_masks': polyline_mask,
        }
        return outputs

def find_best_sperate_plan(idx,array):

    h = array[-1] - array[idx]
    w = idx

    cost  = h*w
    return cost

def get_chunk_idx(polyline_length):
    _polyline_length, polyline_length_idx = torch.sort(polyline_length)

    costs = []
    for i in range(len(_polyline_length)):

        cost = find_best_sperate_plan(i,_polyline_length)
        costs.append(cost)
    seperate_point = torch.stack(costs).argmax()
    chunk1 = polyline_length_idx[:seperate_point+1]
    chunk2 = polyline_length_idx[seperate_point+1:]

    revert_idx = torch.argsort(polyline_length_idx)    

    return chunk1, chunk2, revert_idx, _polyline_length[seperate_point]


def assign_bev(feat, idx):
    return feat[idx]


def assign_batch(batch, idx, size):
    new_batch = {}
    for k,v in batch.items():
        new_batch[k] = v[idx]
        if new_batch[k].ndim > 1:
            new_batch[k] = new_batch[k][:,:size]
    
    return new_batch
