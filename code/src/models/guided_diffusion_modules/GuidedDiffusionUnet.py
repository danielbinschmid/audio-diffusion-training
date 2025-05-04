from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from typing import Literal
from .nn import (
    zero_module,
    normalization,
    gamma_embedding,
)
from typing import List, Set, Tuple
from .components import (
    SiLU,
    EmbedSequential,
    Upsample,
    Downsample,
    ResBlock,
    AttentionBlock,
    next_multiple_of_m,
)
from typing import Optional
from src.models.intermediate_activation_cache import IntermediateActivationCache


class GuidedDiffusionUnetConfig(BaseModel):
    """
    :param in_channel: channels in the input Tensor, for image colorization : Y_channels + X_channels .
    :param inner_channel: base channel count for the model.
    :param out_channel: channels in the output Tensor.
    :param res_blocks: number of residual blocks per downsample.
    :param attn_res: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mults: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    image_size: None
    in_channel: int
    inner_channel: int
    out_channel: int
    res_blocks: int
    attn_res: List | Set | Tuple
    dropout: float = 0
    channel_mults: List[int] = [1, 2, 4, 8]
    conv_resample: bool = True
    use_checkpoint: bool = False
    use_fp16: bool = False
    num_heads: int = 1
    num_head_channels: int = -1
    num_heads_upsample: int = -1
    use_scale_shift_norm: bool = True
    resblock_updown: bool = True
    use_new_attention_order: bool = False
    conditioning_mechanism: Literal["inp_channel", "unconditional"] = "unconditional"
    delta_h_mode: Literal["off", "standard"] = "off"


class GuidedDiffusionUnet(nn.Module):
    """
    The full UNet model with attention and embedding.
    """

    intermediate_activation_cache: IntermediateActivationCache

    def __init__(self, cfg: GuidedDiffusionUnetConfig):
        super().__init__()

        self.intermediate_activation_cache = IntermediateActivationCache()

        # map config to arguments
        self.cfg = cfg
        image_size = cfg.image_size
        in_channel = cfg.in_channel
        inner_channel = cfg.inner_channel
        out_channel = cfg.out_channel
        res_blocks = cfg.res_blocks
        attn_res = cfg.attn_res
        dropout = cfg.dropout
        channel_mults = cfg.channel_mults
        conv_resample = cfg.conv_resample
        use_checkpoint = cfg.use_checkpoint
        use_fp16 = cfg.use_fp16
        num_heads = cfg.num_heads
        num_head_channels = cfg.num_head_channels
        num_heads_upsample = cfg.num_heads_upsample
        use_scale_shift_norm = cfg.use_scale_shift_norm
        resblock_updown = cfg.resblock_updown
        use_new_attention_order = cfg.use_new_attention_order

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channel = in_channel
        self.inner_channel = inner_channel
        self.out_channel = out_channel
        self.res_blocks = res_blocks
        self.attn_res = attn_res
        self.dropout = dropout
        self.channel_mults = channel_mults
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        cond_embed_dim = inner_channel * 4
        self.cond_embed = nn.Sequential(
            nn.Linear(inner_channel, cond_embed_dim),
            SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )

        ch = input_ch = int(channel_mults[0] * inner_channel)
        self.input_blocks = nn.ModuleList(
            [EmbedSequential(nn.Conv2d(in_channel, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        self.downsample_size: int = 1
        for level, mult in enumerate(channel_mults):
            for _ in range(res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        cond_embed_dim,
                        dropout,
                        out_channel=int(mult * inner_channel),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * inner_channel)
                if ds in attn_res:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mults) - 1:
                out_ch = ch
                self.input_blocks.append(
                    EmbedSequential(
                        ResBlock(
                            ch,
                            cond_embed_dim,
                            dropout,
                            out_channel=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, out_channel=out_ch)
                    )
                )
                self.downsample_size = 2 * self.downsample_size

                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = EmbedSequential(
            ResBlock(
                ch,
                cond_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                cond_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mults))[::-1]:
            for i in range(res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        cond_embed_dim,
                        dropout,
                        out_channel=int(inner_channel * mult),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(inner_channel * mult)
                if ds in attn_res:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            cond_embed_dim,
                            dropout,
                            out_channel=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, out_channel=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(nn.Conv2d(input_ch, out_channel, 3, padding=1)),
        )

    def forward(
        self,
        x: torch.Tensor,
        gammas,
        class_label: torch.Tensor,
        cache_intermediate_activations: bool = False,
        down_block_additional_residuals: Optional[List[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
    ):
        """
        Apply the model to an input batch.
        :param x: an [N x 2 x ...] Tensor of inputs (B&W)
        :param gammas: a 1-D batch of gammas.
        :param class_label: [N x E] Tensor of class label embeddings where E is the embedding dimension
        :return: an [N x C x ...] Tensor of outputs.
        """
        padding_val = next_multiple_of_m(x.shape[3], self.downsample_size) - x.shape[3]

        x = F.pad(x, pad=(0, padding_val, 0, 0))

        if self.cfg.conditioning_mechanism == "inp_channel":
            assert class_label is not None

            # concatenate conditioning tensor to input channel as in https://huggingface.co/learn/diffusion-course/unit2/3
            bs, _, w, h = x.shape
            cls_emb_dim = class_label.shape[1]
            class_label = class_label.view(bs, cls_emb_dim, 1, 1).expand(
                bs, cls_emb_dim, w, h
            )
            x = torch.cat((x, class_label), 1)

        hs = []
        gammas = gammas.view(-1,)
        emb = self.cond_embed(gamma_embedding(gammas, self.inner_channel))

        h = x.type(torch.float32)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h: torch.Tensor = self.middle_block(h, emb)

        if cache_intermediate_activations:
            self.intermediate_activation_cache.add_hspace_sample(h_sample=h, t=gammas)

        if mid_block_additional_residual is not None:
            h = h + mid_block_additional_residual

        for module in self.output_blocks:
            downblock_residual = hs.pop()
            if down_block_additional_residuals is not None:
                downblock_additional_residual = down_block_additional_residuals.pop()
                downblock_residual = downblock_residual + downblock_additional_residual
            h = torch.cat([h, downblock_residual], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)[:, :, :, :-padding_val] if padding_val > 0 else self.out(h)

    @staticmethod
    def get_argument_of_this_model() -> dict:
        from HParams import HParams

        h_params = HParams()
        model_argument: dict = getattr(h_params.model, "GuidedDiffusionUnet", dict())
        return model_argument


if __name__ == "__main__":
    b, c, h, w = 3, 6, 64, 64
    timsteps = 100
    model = GuidedDiffusionUnet(
        image_size=h,
        in_channel=c,
        inner_channel=64,
        out_channel=3,
        res_blocks=2,
        attn_res=[8],
    )
    x = torch.randn((b, c, h, w))
    emb = torch.ones((b,))
    out = model(x, emb)
