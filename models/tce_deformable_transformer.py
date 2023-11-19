# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn
# from models.efficient_attention import EfficientAttention

from einops import rearrange, repeat

# 392192 to 65535

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300, q_trans = False, tdetr = False, f_token = 0):
        super().__init__()
        print("+++++++++++++++++++++++using updated deformable detr++++++++++++++++++++++++++++")
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_feature_level = num_feature_levels

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels,
                                                          nhead, enc_n_points, tdetr, f_token)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers, f_token = f_token)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                        dropout, activation,
                                                        num_feature_levels, 
                                                        nhead, dec_n_points, is_query_atten = q_trans)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2) # reference point here (x, y)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, tgt, masks, pos_embeds, query_embed=None):
        assert self.two_stage or query_embed is not None
        """
        srcs (list[Tensor]): list of tensors num_layers x [batch_size*time, c, hi, wi], input of encoder
        tgt (Tensor): [batch_size, time, c, num_queries_per_frame]
        masks (list[Tensor]): list of tensors num_layers x [batch_size*time, hi, wi], the mask of srcs
        pos_embeds (list[Tensor]): list of tensors num_layers x [batch_size*time, c, hi, wi], position encoding of srcs
        query_embed (Tensor): [num_queries, c]
        """
        # prepare input for encoder
        bs, C, H, W = srcs[-1].shape
        # tgt_srcs = repeat(tgt, 'b c -> t b w c', t=bs, w=1)
        # tgt_srcs = rearrange(tgt_srcs, 't b w c -> t c b w')
        # tgt_mask = torch.zeros(bs,1,1, device = masks[0].device, dtype = torch.bool)
        # tgt_pos = torch.zeros(bs,C, 1,1,device = pos_embeds[0].device)
        # # print(tgt_srcs.shape)
        # # assert False
        # srcs.append(tgt_srcs)
        # masks.append(tgt_mask)
        # pos_embeds.append(tgt_pos)

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2) # [batch_size, hi*wi, c]

            mask = mask.flatten(1)               # [batch_size, hi*wi]
            pos_embed = pos_embed.flatten(2).transpose(1, 2) # [batch_size, hi*wi, c]
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)

            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        # For a clip, concat all the features, first fpn layer size, then frame size
        src_flatten = torch.cat(src_flatten, 1)     # [bs*t, \sigma(hi*wi), c] 
        mask_flatten = torch.cat(mask_flatten, 1)   # [bs*t, \sigma(hi*wi)]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) 
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        # memory: [bs*t, \sigma(hi*wi), c]
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        
        # prepare input for decoder
        bs, _, c = memory.shape 
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            # tgt = repeat(tgt, 'b c -> b t q c', t=bs, q=query_embed.shape[0])
            # query_embed = rearrange(query_embed, '(t q) c -> t q c', t=bs)
            # tgt = repeat(tgt, 'b c -> b t q c', t=bs, q=query_embed.shape[1])
            # print(tgt.shape)
            # assert False
            b, t, q, c = tgt.shape
            tgt = rearrange(tgt, 'b t q c -> (b t) q c')
            
            # query_embed = query_embed.expand(b*t, -1, -1)      # [batch_size*time, num_queries_per_frame, c]
            query_embed = query_embed.unsqueeze(0).expand(b*t, -1, -1)      # [batch_size*time, num_queries_per_frame, c]
            reference_points = self.reference_points(query_embed).sigmoid() # [batch_size*time, num_queries_per_frame, 2]
            init_reference_out = reference_points

        # decoder
        hs, inter_references, inter_samples = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)

        inter_references_out = inter_references

        # convert memory to fpn format
        memory_features = []  # 8x -> 32x
        spatial_index = 0
        for lvl in range(self.num_feature_level - 1):
            h, w = spatial_shapes[lvl]
            
            # [bs*t, c, h, w]
            memory_lvl = memory[:, spatial_index : spatial_index + h * w, :].reshape(bs, h, w, c).permute(0, 3, 1, 2).contiguous()  
            memory_features.append(memory_lvl)
            spatial_index += h * w

        if self.two_stage:
            return hs, memory_features, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact, inter_samples
        # hs: [l, batch_size*time, num_queries_per_frame, c], where l is number of decoder layers
        # init_reference_out: [batch_size*time, num_queries_per_frame, 2]
        # inter_references_out: [l, batch_size*time, num_queries_per_frame, 4]
        # memory: [batch_size*time, \sigma(hi*wi), c]
        # memory_features: list[Tensor]

        return hs, memory_features, init_reference_out, inter_references_out, memory, None, inter_samples

# The model for the cross frame connection
class CrossFrameConn(nn.Module):
    def __init__(self, init_value=1.0, direction = 0, d_model = 256, skip_frame = False):
        '''
        direction: 0 for only left frame, 1 for only right frame, 2 for two frames
        '''
        super(CrossFrameConn, self).__init__()

        self.direction = direction
        self.scale1 = nn.Parameter(torch.FloatTensor([init_value]))
        
        if self.direction > 1:
            self.scale2 = nn.Parameter(torch.FloatTensor([init_value]))

        self.norm = nn.LayerNorm(d_model)
        self.skip_frame = skip_frame

    def forward(self, src):
        '''
        feature aggregates between two frames
        Input:
            src: features for the current frame
        '''
        # generate the tensor for previous frame

        if self.direction == 0:
            pre_src = _rotate_tensor(src, bs = 1, step = 1)
            output = self.scale1 * pre_src + src
            output = self.norm(output)
        elif self.direction == 1:
            lat_src = _rotate_tensor(src, bs = 1, step = -1)
            output = self.scale1 * lat_src + src
            output = self.norm(output)
        else:
            pre_src = _rotate_tensor(src, bs = 1, step = 1)
            lat_src = _rotate_tensor(src, bs = 1, step = -1)
            output = self.scale1 * pre_src + self.scale2 * lat_src + src
            output = self.norm(output)
        return output

class QueryTransLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, self_atten = True):
        super().__init__()

        self.self_atten = self_atten
        # cross attention
        if self.self_atten:
        
            self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
            self.dropout1 = nn.Dropout(dropout)
            self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        if self.self_atten:
            tgt2, sampling_locations, attention_weights = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                                reference_points,
                                src, src_spatial_shapes, level_start_index, src_padding_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt, sampling_locations, attention_weights

class LastLayerAsToken(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu", n_heads=8) -> None:
        super().__init__()

        self.inter_frame_att = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, level_start_index):
        
        src_dense = src[:,:level_start_index[-1],:]
        src_token = src[:,level_start_index[-1]:,:]
        pos_token = pos[:,level_start_index[-1]:,:]
        # src_token = self.with_pos_embed(src_token, pos_token)
        t,q,c = src_token.shape
        src_token = rearrange(src_token, 't q c -> (t q) c').unsqueeze(1)
        pos_token = rearrange(pos_token, 't q c -> (t q) c').unsqueeze(1)
        src_token2 = self.inter_frame_att(self.with_pos_embed(src_token, pos_token), src_token, src_token)[0]
        src_token = src_token + self.dropout1(src_token2)
        src_token = self.forward_ffn(src_token)
        src_token = rearrange(src_token.squeeze(1), '(t q) c -> t q c', t = t, q = q)
        src = torch.cat((src_dense, src_token), dim = 1)
        return src

class FrameTokenLayer(nn.Module):
    '''
    The model layer for the encoder to first generate tokens for each frame and then
    communicate between tokens. Finally 
    '''
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu", n_heads=8, n_levels=4, n_points=4) -> None:
        super().__init__()

        self.reference_points = nn.Linear(d_model, 2)

        # token get info from frames
        self.token_frame_atten = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # token level communication
        self.token_self_atten = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # update frame features with token
        self.frame_token_atten = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # # a linear layer to generate the token info output
        # self.linear3 = nn.Linear(d_model, d_ffn)
        # self.linear4 = nn.Linear(d_ffn, d_model)
        # self.dropout6 = nn.Dropout(dropout)
        # self.dropout7 = nn.Dropout(dropout)
        # self.norm5 = nn.LayerNorm(d_model)

        # # a linear layer to generate the token self atten info output
        # self.linear5 = nn.Linear(d_model, d_ffn)
        # self.linear6 = nn.Linear(d_ffn, d_model)
        # self.dropout8 = nn.Dropout(dropout)
        # self.dropout9 = nn.Dropout(dropout)
        # self.norm6 = nn.LayerNorm(d_model)

        # a linear layer to generate the final output
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)
        self.norm4 = nn.LayerNorm(d_model)



    def _reset_parameters(self):
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos, token, token_pose, src_spatial_shapes, level_start_index, src_padding_mask, valid_ratios):

        B,To,C = token.shape
        # Cross atten token with src to get info from each frame.
        ref_point = self.reference_points(token).sigmoid()
        ref_point = ref_point[:, :, None] * valid_ratios[:, None]
        
    
        token2, sampling_locations, attention_weights = self.token_frame_atten(self.with_pos_embed(token, token_pose),
                               ref_point, src, src_spatial_shapes, level_start_index, src_padding_mask)
        token = token + self.dropout1(token2)
        token = self.norm1(token)

        # # linear layer
        # token2 = self.linear4(self.dropout6(self.activation(self.linear3(token))))
        # token = token + self.dropout7(token2)
        # token = self.norm5(token)


        # Self atten between all tokens to communicate between frames
        token = rearrange(token, "b t c -> (b t) c").unsqueeze(1) #[b*t, 1, c]
        token_pose1 = rearrange(token_pose, "b t c -> (b t) c").unsqueeze(1)
        q = k = self.with_pos_embed(token, token_pose1)
        
        token2 = self.token_self_atten(q, k, token)[0]
        token = token + self.dropout2(token2)
        token = self.norm2(token)

        # # linear layer
        # token2 = self.linear6(self.dropout8(self.activation(self.linear5(token))))
        # token = token + self.dropout9(token2)
        # token = self.norm6(token)


        token = rearrange(token.squeeze(1), '(b t) c -> b t c', b=B)

        # Update the frame info with token
        q = self.with_pos_embed(src, pos)
        k = self.with_pos_embed(token, token_pose)
        src2 = self.frame_token_atten(q.transpose(0, 1), k.transpose(0, 1), token.transpose(0, 1))[0].transpose(0, 1)
        src = src + self.dropout3(src2)
        src = self.norm3(src)



        # Linear layers for output
        src2 = self.linear2(self.dropout4(self.activation(self.linear1(src))))
        src = src + self.dropout5(src2)
        src = self.norm4(src)

        return src, token

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, tdetr = False, f_token = 0):
        super().__init__()

        if f_token > 0:
            self.ftoken_layers = FrameTokenLayer(d_model, d_ffn,
                 dropout, activation, n_heads, n_levels, n_points)

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
            
        self.f_token = f_token
        if f_token < 0:
            print("Using last feature layer as inter frame token for encoder.")
            self.inter_frame_atten = LastLayerAsToken(d_model, d_ffn, dropout, activation, n_heads)
        # self.self_attn = EfficientAttention(d_model, d_model, n_heads, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, valid_ratios, padding_mask=None, memory_bus = None, memory_pos = None):

        if self.f_token < 0:
            src = self.inter_frame_atten(src,pos,level_start_index)
        if self.f_token > 0:
            assert (memory_bus is not None) and (memory_pos is not None)
            src, memory_bus = self.ftoken_layers(src, pos, memory_bus, memory_pos, spatial_shapes, level_start_index, padding_mask, valid_ratios)

        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, 
                                                                src, spatial_shapes, level_start_index, padding_mask)[0]
        # src2 = self.self_attn(self.with_pos_embed(src, pos))

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src, memory_bus


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, d_model = 256, skip_frame = False, f_token = 0):
        super().__init__()
        self.f_token = f_token
        if self.f_token > 0:
            print(f"Using {f_token} frame tokens for each frame.")
            self.memory_bus = torch.nn.Parameter(torch.randn(f_token, d_model), requires_grad=True)
            self.memory_pos = torch.nn.Parameter(torch.randn(f_token, d_model), requires_grad=True)
            nn.init.kaiming_normal_(self.memory_bus, mode="fan_out", nonlinearity="relu")
            nn.init.kaiming_normal_(self.memory_pos, mode="fan_out", nonlinearity="relu")
            
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        # self.cross_frame_connection = _get_clones(CrossFrameConn(direction = 0, d_model = d_model, skip_frame = skip_frame), num_layers - 1)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # meshgrid generates grid for location
            # linspace(start, end, total_number) generates a list with same distance
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            # reshape to 1D
            # valid ratios is the valid position/all position, valid pos means not masked bu padding mask
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            # print(ref_y)
            # assert False
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    # @staticmethod
    # def get_reference_points_3d(spatial_shapes, valid_ratios, device):
    #     reference_points_list = []
    #     for lvl, (H_, W_) in enumerate(spatial_shapes):
    #         # meshgrid generates grid for location
    #         # linspace(start, end, total_number) generates a list with same distance
    #         ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
    #                                       torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            
    #         # reshape to 1D
    #         # valid ratios is the valid position/all position, valid pos means not masked bu padding mask
    #         ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
    #         ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
    #         ref = torch.stack((ref_x, ref_y), -1)
    #         reference_points_list.append(ref)
    #     reference_points = torch.cat(reference_points_list, 1)
    #     reference_points = reference_points[:, :, None] * valid_ratios[:, None]

    #     return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)

        if self.f_token > 0:
            B,_,_ = output.shape
            memory_bus = self.memory_bus[None,:,:].repeat(B,1,1)
            memory_pos = self.memory_pos[None,:,:].repeat(B,1,1)
        
        for lvl, layer in enumerate(self.layers):
            if self.f_token > 0:
                output, memory_bus = layer(output, pos, reference_points, spatial_shapes, level_start_index, valid_ratios, 
                                        padding_mask, memory_bus, memory_pos)
            else:
                output,_ = layer(output, pos, reference_points, spatial_shapes, level_start_index, valid_ratios, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, is_query_atten = False):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # if is_query_atten:
        #     # self attention query base
        #     self.self_attn2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        #     self.dropout5 = nn.Dropout(dropout)
        #     self.norm4 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # Config
        self.is_query_atten = is_query_atten
        if self.is_query_atten:
            print("Using query trans layer for decoder.")

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        if self.is_query_atten:
            # tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
            # tgt = tgt + self.dropout5(tgt2)
            # tgt = self.norm4(tgt)
            # tgt2 = self.self_attn2(q, k, tgt)[0]
            tgt2 = self.self_attn(q, k, tgt)[0]
        else:
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2, sampling_locations, attention_weights = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt, sampling_locations, attention_weights



class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, d_model=256, skip_frame = False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

        # Skip one frame for training
        self.skip_frame = skip_frame

        # Cross frame connection
        
        # self.cross_frame_connection = _get_clones(CrossFrameConn(direction = 0, d_model = d_model, skip_frame = skip_frame), num_layers - 1)
        # self.query_trans = _get_clones(QueryTransLayer(), self.num_layers-1)

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        # we modify here for get the information of sample points
        output = tgt
        # if self.training and self.skip_frame:
        #     # frame num in each clip is 5
        #     # Randomly select one frame to be skiped
        #     skip_frame_id = torch.rand(1)
        #     skip_frame_id = int(skip_frame_id[0] * 6 // 1)
        #     if skip_frame_id <= 4.5:
        #         src_b, src_h, src_w = src.shape
        #         src[skip_frame_id] = torch.zeros(src_h, src_w)
        #         src  = src*5/4


        intermediate = []
        intermediate_reference_points = []
        intermediate_samples = [] # sample points
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output, sampling_locations, attention_weights = layer(output, query_pos, reference_points_input, 
                                                        src, src_spatial_shapes, src_level_start_index, src_padding_mask)
            # sampling_loactions: [N, Len_q, self.n_heads, self.n_levels, self.n_points, 2], 
            #                     [B, Q, n_head, n_level(num_feature_level*num_frames), n_points, 2]
            # attention_weights: [B, Q, n_head, n_level(num_feature_level*num_frames), n_points]
            # src_valid_ratios: [N, self.n_levels, 2]
            N, Len_q = sampling_locations.shape[:2]
            sampling_locations = sampling_locations / src_valid_ratios[:, None, None, :, None, :]
            weights_flat = attention_weights.view(N, Len_q, -1)      # [B, Q, n_head * n_level * n_points]
            samples_flat = sampling_locations.view(N, Len_q, -1, 2)  # [B, Q, n_head * n_level * n_points, 2]
            top_weights, top_idx = weights_flat.topk(30, dim=2)      # [B, Q, 30], [B, Q, 30]
            weights_keep = torch.gather(weights_flat, 2, top_idx)    # [B, Q, 30]
            samples_keep = torch.gather(samples_flat, 2, top_idx.unsqueeze(-1).repeat(1, 1, 1, 2))  # [B, Q, 30, 2]

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_samples.append(samples_keep)

            # cross frame connection
            # if lid < num_layers - 1:
            #     # print(output)
            #     output = self.cross_frame_connection[lid](output)
                # output, sampling_locations, attention_weights = self.query_trans[lid // 2](output, query_pos, reference_points_input, 
                                                        # src, src_spatial_shapes, src_level_start_index, src_padding_mask)
                # print(output)


        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), torch.stack(intermediate_samples)

        return output, reference_points, samples_keep


class SentenceRefine(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=0.0):
        super().__init__()
        multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.layers = _get_clones(multihead_attn, num_layers)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, layer,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt2 = self.layers[layer](query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt * tgt2
        return tgt


def _rotate_tensor(src, bs:int, step:int = 1):
    '''
    rotate the features in each video
    '''
    bt,_,_ = src.shape
    assert bt//bs*bs == bt, "the batch size is incorrect."
    t = bt//bs
    # while(step<0):
    #     step += t
    step = step%t
    # print("step",step,"t",t)
    res = None
    for batch in range(bt):
        temp = src[batch*t : (batch+1)*t]
        rot = torch.cat((temp[step:],temp[:step]),dim = 0)
        if res is None:
            res = rot
        else:
            res = torch.cat((res,rot),dim = 0)
    
    return res

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _gen_qtrans(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points, num_decoder_layers):
    model = nn.ModuleList([])
    for idx in range(num_decoder_layers):
        if idx % 2 == 0:
            model.append(DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, 
                                                            nhead, dec_n_points, is_query_atten = False))
        else:
            model.append(DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, 
                                                            nhead, dec_n_points, is_query_atten = True))
    return model

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        q_trans = args.qtrans, f_token = args.f_token
        )
