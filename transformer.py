"""
Adapted from: https://github.com/openai/openai/blob/55363aa496049423c37124b440e9e30366db3ed6/orc/orc/diffusion/vit.py
"""


import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from models.Transformer_utils import *
from .checkpoint import checkpoint
from .pretrained_clip import FrozenImageCLIP, ImageCLIP, ImageType
from .util import timestep_embedding


def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        self.attention = QKVMultiheadAttention(device=device, dtype=dtype, heads=heads, n_ctx=n_ctx)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        x = checkpoint(self.attention, (x,), (), True)
        x = self.c_proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, width: int, init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width * 4, width, device=device, dtype=dtype)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, heads: int, n_ctx: int):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.heads = heads
        self.n_ctx = n_ctx

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    device=device,
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x


class PointDiffusionTransformer(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        input_channels: int = 3,
        output_channels: int = 3,
        n_ctx: int = 1024,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        init_scale: float = 0.25,
        time_token_cond: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_ctx = n_ctx
        self.time_token_cond = time_token_cond
        self.time_embed = MLP(
            device=device, dtype=dtype, width=width, init_scale=init_scale * math.sqrt(1.0 / width)
        )
        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        self.backbone = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx + int(time_token_cond),
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.input_proj = nn.Linear(input_channels, width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, output_channels, device=device, dtype=dtype)
        self.mix=nn.Linear(1024, 512, device=device, dtype=dtype)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :return: an [N x C' x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        return self._forward_with_cond(x, [(t_embed, self.time_token_cond)])

    def _forward_with_cond(
        self, pe: torch.Tensor,x_partial: torch.Tensor, x: torch.Tensor, cond_as_token: List[Tuple[torch.Tensor, bool]]
    ) -> torch.Tensor:
        # print([x.shape,self.input_channels])
        h = self.input_proj(x.permute(0, 2, 1))  # NCL -> NLC
        
        # print([h.shape,partial.shape])
        # print([x.permute(0, 2, 1).shape,h.shape])
        # h=self.mix(torch.cat((h, partial), dim=2))
        # h+=partial
        # print([x.permute(0, 2, 1).shape,h.shape])
        for emb, as_token in cond_as_token:
            if not as_token:
                h = h + emb[:, None]
        extra_tokens = [
            (emb[:, None] if len(emb.shape) == 2 else emb)
            for emb, as_token in cond_as_token
            if as_token
        ] # extra_tokens[0]: [bs,1,512]
        # print(extra_tokens[0].shape)
        if len(extra_tokens):
            h = torch.cat(extra_tokens + [h], dim=1) #[bs, 1024, 512] -> [bs, 1026, 512]
        h=torch.cat([(x_partial+pe),h], dim=1)
        # print(cond_as_token)
        
        h = self.ln_pre(h)
        # print(h.shape)
        h = self.backbone(h)
        # print(h.shape)
        h = self.ln_post(h)
        if len(extra_tokens):
            h = h[:, sum(h.shape[1] for h in extra_tokens)+256:]
        h = self.output_proj(h)
        return h.permute(0, 2, 1)


class CLIPImagePointDiffusionTransformer(PointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int = 1024,
        token_cond: bool = False,
        cond_drop_prob: float = 0.0,
        frozen_clip: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(device=device, dtype=dtype, n_ctx=n_ctx + int(token_cond), **kwargs)
        self.n_ctx = n_ctx
        self.token_cond = token_cond
        self.clip = (FrozenImageCLIP if frozen_clip else ImageCLIP)(device, cache_dir=cache_dir)

        self.clip_embed = nn.Linear(
            self.clip.feature_dim, self.backbone.width, device=device, dtype=dtype
        )
        self.cond_drop_prob = cond_drop_prob
        # self.ln=nn.Linear(3, 512)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, 512)
        )  
        self.center_num  = [512,256]
        self.grouper = DGCNN_Grouper(k = 16)
        self.input_proj_ = nn.Sequential(
            nn.Linear(self.grouper.num_features, 512),
            nn.GELU(),
            nn.Linear(512, 512)
        )
    def cached_model_kwargs(self, batch_size: int, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        with torch.no_grad():
            return dict(embeddings=self.clip(batch_size, **model_kwargs))

    def forward(
        self,
        partial: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        images: Optional[Iterable[Optional[ImageType]]] = None,
        texts: Optional[Iterable[Optional[str]]] = None,
        embeddings: Optional[Iterable[Optional[torch.Tensor]]] = None,
    ):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :param images: a batch of images to condition on.
        :param texts: a batch of texts to condition on.
        :param embeddings: a batch of CLIP embeddings to condition on.
        :return: an [N x C' x T] tensor.
        """
        # print([x.shape[-1],self.n_ctx])
        coor, f = self.grouper(partial, self.center_num) # b n c
         
        pe =  self.pos_embed(coor)
        x_partial = self.input_proj_(f)
        # print([pe.shape,x.shape])
        # partial=self.ln(partial)
        assert x.shape[-1] == self.n_ctx

        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        clip_out = self.clip(batch_size=len(x), images=images, texts=texts, embeddings=embeddings)
        assert len(clip_out.shape) == 2 and clip_out.shape[0] == x.shape[0]
        #条件丢弃（conditional dropout）的正则化技术
        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            clip_out = clip_out * mask[:, None].to(clip_out)

        # Rescale the features to have unit variance
        clip_out = math.sqrt(clip_out.shape[1]) * clip_out

        clip_embed = self.clip_embed(clip_out)
        # print([clip_out.shape,clip_embed.shape])
        cond = [(clip_embed, self.token_cond), (t_embed, self.time_token_cond)]
        return self._forward_with_cond(pe,x_partial,x, cond)


class CLIPImageGridPointDiffusionTransformer(PointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int = 1024,
        cond_drop_prob: float = 0.0,
        frozen_clip: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        clip = (FrozenImageCLIP if frozen_clip else ImageCLIP)(
            device,
            cache_dir=cache_dir,
        )
        super().__init__(device=device, dtype=dtype, n_ctx=n_ctx + clip.grid_size**2, **kwargs)
        self.n_ctx = n_ctx
        self.clip = clip
        self.clip_embed = nn.Sequential(
            nn.LayerNorm(
                normalized_shape=(self.clip.grid_feature_dim,), device=device, dtype=dtype
            ),
            nn.Linear(self.clip.grid_feature_dim, self.backbone.width, device=device, dtype=dtype),
        )
        self.cond_drop_prob = cond_drop_prob

    def cached_model_kwargs(self, batch_size: int, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        _ = batch_size
        with torch.no_grad():
            return dict(embeddings=self.clip.embed_images_grid(model_kwargs["images"]))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        images: Optional[Iterable[ImageType]] = None,
        embeddings: Optional[Iterable[torch.Tensor]] = None,
    ):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :param images: a batch of images to condition on.
        :param embeddings: a batch of CLIP latent grids to condition on.
        :return: an [N x C' x T] tensor.
        """
        assert images is not None or embeddings is not None, "must specify images or embeddings"
        assert images is None or embeddings is None, "cannot specify both images and embeddings"
        assert x.shape[-1] == self.n_ctx

        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))

        if images is not None:
            clip_out = self.clip.embed_images_grid(images)
        else:
            clip_out = embeddings

        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            clip_out = clip_out * mask[:, None, None].to(clip_out)

        clip_out = clip_out.permute(0, 2, 1)  # NCL -> NLC
        clip_embed = self.clip_embed(clip_out)

        cond = [(t_embed, self.time_token_cond), (clip_embed, True)]
        return self._forward_with_cond(x, cond)


class UpsamplePointDiffusionTransformer(PointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        cond_input_channels: Optional[int] = None,
        cond_ctx: int = 1024,
        n_ctx: int = 4096 - 1024,
        channel_scales: Optional[Sequence[float]] = None,
        channel_biases: Optional[Sequence[float]] = None,
        **kwargs,
    ):
        super().__init__(device=device, dtype=dtype, n_ctx=n_ctx + cond_ctx, **kwargs)
        self.n_ctx = n_ctx
        self.cond_input_channels = cond_input_channels or self.input_channels
        self.cond_point_proj = nn.Linear(
            self.cond_input_channels, self.backbone.width, device=device, dtype=dtype
        )

        self.register_buffer(
            "channel_scales",
            torch.tensor(channel_scales, dtype=dtype, device=device)
            if channel_scales is not None
            else None,
        )
        self.register_buffer(
            "channel_biases",
            torch.tensor(channel_biases, dtype=dtype, device=device)
            if channel_biases is not None
            else None,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, *, low_res: torch.Tensor):
        """
        :param x: an [N x C1 x T] tensor.
        :param t: an [N] tensor.
        :param low_res: an [N x C2 x T'] tensor of conditioning points.
        :return: an [N x C3 x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        low_res_embed = self._embed_low_res(low_res)
        cond = [(t_embed, self.time_token_cond), (low_res_embed, True)]
        return self._forward_with_cond(x, cond)

    def _embed_low_res(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_scales is not None:
            x = x * self.channel_scales[None, :, None]
        if self.channel_biases is not None:
            x = x + self.channel_biases[None, :, None]
        return self.cond_point_proj(x.permute(0, 2, 1))


class CLIPImageGridUpsamplePointDiffusionTransformer(UpsamplePointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int = 4096 - 1024,
        cond_drop_prob: float = 0.0,
        frozen_clip: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        clip = (FrozenImageCLIP if frozen_clip else ImageCLIP)(
            device,
            cache_dir=cache_dir,
        )
        super().__init__(device=device, dtype=dtype, n_ctx=n_ctx + clip.grid_size**2, **kwargs)
        self.n_ctx = n_ctx

        self.clip = clip
        self.clip_embed = nn.Sequential(
            nn.LayerNorm(
                normalized_shape=(self.clip.grid_feature_dim,), device=device, dtype=dtype
            ),
            nn.Linear(self.clip.grid_feature_dim, self.backbone.width, device=device, dtype=dtype),
        )
        self.cond_drop_prob = cond_drop_prob

    def cached_model_kwargs(self, batch_size: int, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if "images" not in model_kwargs:
            zero_emb = torch.zeros(
                [batch_size, self.clip.grid_feature_dim, self.clip.grid_size**2],
                device=next(self.parameters()).device,
            )
            return dict(embeddings=zero_emb, low_res=model_kwargs["low_res"])
        with torch.no_grad():
            return dict(
                embeddings=self.clip.embed_images_grid(model_kwargs["images"]),
                low_res=model_kwargs["low_res"],
            )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        low_res: torch.Tensor,
        images: Optional[Iterable[ImageType]] = None,
        embeddings: Optional[Iterable[torch.Tensor]] = None,
    ):
        """
        :param x: an [N x C1 x T] tensor.
        :param t: an [N] tensor.
        :param low_res: an [N x C2 x T'] tensor of conditioning points.
        :param images: a batch of images to condition on.
        :param embeddings: a batch of CLIP latent grids to condition on.
        :return: an [N x C3 x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        low_res_embed = self._embed_low_res(low_res)

        if images is not None:
            clip_out = self.clip.embed_images_grid(images)
        elif embeddings is not None:
            clip_out = embeddings
        else:
            # Support unconditional generation.
            clip_out = torch.zeros(
                [len(x), self.clip.grid_feature_dim, self.clip.grid_size**2],
                dtype=x.dtype,
                device=x.device,
            )

        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            clip_out = clip_out * mask[:, None, None].to(clip_out)

        clip_out = clip_out.permute(0, 2, 1)  # NCL -> NLC
        clip_embed = self.clip_embed(clip_out)

        cond = [(t_embed, self.time_token_cond), (clip_embed, True), (low_res_embed, True)]
        return self._forward_with_cond(x, cond)
    
    # class PartialCond(nn.Module):
    #     def __init__(
    #         self,
    #         inChannel,
    #         outChannel=512
    #     ):
    #         super().__init__()
    #         self.inChannel=inChannel
    #         self.outChannel=outChannel
    #         self.ln=nn.Linear(inChannel, outChannel)

    # def forward(self, x):
    #     return self.ln(x)
class DGCNN_Grouper(nn.Module):
    def __init__(self, k = 16):
        super().__init__()
        '''
        K has to be 16
        '''
        print('using group version 2')
        self.k = k
        # self.knn = KNN(k=k, transpose_mode=False)
        self.input_trans = nn.Conv1d(3, 8, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 32),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 128),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )
        self.num_features = 128
    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            # _, idx = self.knn(coor_k, coor_q)  # bs k np
            idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous()) # B G M
            idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, x, num):
        '''
            INPUT:
                x : bs N 3
                num : list e.g.[1024, 512]
            ----------------------
            OUTPUT:

                coor bs N 3
                f    bs N C(128) 
        '''
        x = x.transpose(-1, -2).contiguous()

        coor = x
        f = self.input_trans(x)

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, num[0])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, num[1])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        coor = coor.transpose(-1, -2).contiguous()
        f = f.transpose(-1, -2).contiguous()

        return coor, f