from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.models.attention import Attention

class AdaGroupNorm(nn.Module):
    def __init__(
        self, embedding_dim: int, out_dim: int, num_groups: int, act_fn: Optional[bool] = False, eps: float = 1e-5
    ):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

        if act_fn:
            self.act = nn.SiLU()
        else:
            self.act = None

        self.linear = nn.Linear(embedding_dim, out_dim * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.act:
            emb = self.act(emb)
        emb = self.linear(emb)
        emb = emb[:, :, None, None]
        scale, shift = emb.chunk(2, dim=1)

        x = F.group_norm(x, self.num_groups, eps=self.eps)
        x = x * (1 + scale) + shift
        return x

class ResnetBlockCondNorm2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        eps: float = 1e-6,
        output_scale_factor: float = 1.0,
        conv_shortcut_bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.output_scale_factor = output_scale_factor

        conv_cls = nn.Conv2d

        self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
        self.conv1 = conv_cls(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups, eps=eps)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = conv_cls(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.in_channels != out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = conv_cls(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(self, input_tensor: torch.FloatTensor, temb: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states, temb)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states, temb)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        return output_tensor
    

class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        eps: float = 1e-6,
        time_embedding_norm: str = "default",  # default, scale_shift,
        output_scale_factor: float = 1.0,
        conv_shortcut_bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm

        linear_cls = nn.Linear
        conv_cls = nn.Conv2d

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = conv_cls(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = linear_cls(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = linear_cls(temb_channels, 2 * out_channels)
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = conv_cls(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.in_channels != out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = conv_cls(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(self, input_tensor: torch.FloatTensor, temb: torch.FloatTensor = None) -> torch.FloatTensor:
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None and temb is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError(
                    f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                )
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        return output_tensor

# ------------------------------------
    
class Downsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        kernel_size: int = 3,
        norm: bool = False,
        eps: float = 1e-6,
        elementwise_affine: bool = None,
        bias: bool = True
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        conv_cls = nn.Conv2d

        if norm is None:
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        else:
            self.norm = None

        if use_conv:
            self.conv = conv_cls(
                self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )
        else:
            assert self.channels == self.out_channels
            self.conv = nn.AvgPool2d(kernel_size=stride, stride=stride)


    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)
        return hidden_states
    

class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int = 512,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default", # or "scale_shift"
        resnet_groups: int = 32,
        output_scale_factor: float = 1.0,
        downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    output_scale_factor=output_scale_factor
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if downsample:
            self.downsampler = Downsample2D(
                out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding
            )
        else:
            self.downsampler = None

    def forward(self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states)

        return hidden_states
    
# ------------------------------------
    
class Upsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        kernel_size: Optional[int] = None,
        padding: int = 1,
        norm: bool = False,
        eps: float = 1e-6,
        elementwise_affine = None,
        bias: bool = True,
        interpolate: bool = True
    ):
        super().__init__()
        assert not (use_conv and use_conv_transpose), "Only one of `use_conv` and `use_conv_transpose` can be True"
        assert not (interpolate and use_conv_transpose), "interpolate and use_conv_transpose cannot be True at the same time"
        assert (use_conv_transpose or interpolate), "At least one of `use_conv_transpose` and `interpolate` should be True"

        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.interpolate = interpolate
        conv_cls = nn.Conv2d

        if norm:
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)

        else:
            self.norm = None

        self.conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            self.conv = nn.ConvTranspose2d(
                channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            self.conv = conv_cls(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)


    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if self.interpolate:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        if self.use_conv:
            hidden_states = self.conv(hidden_states)

        return hidden_states


    
class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int = 512,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # "scale_shift"
        resnet_groups: int = 32,
        output_scale_factor: float = 1.0,
        upsample: bool = True
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    output_scale_factor=output_scale_factor
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if upsample:
            self.upsampler = Upsample2D(out_channels, use_conv=True, out_channels=out_channels)
        else:
            self.upsampler = None


    def forward(self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states)

        return hidden_states

    
# ------------------------------------
    
class UNetMidBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_groups: int = 32,
        add_attention: bool = True,
        attention_num_heads: int = 8,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                output_scale_factor=output_scale_factor
            )
        ]
        attentions = []

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        spatial_dims=2,
                        in_channels=in_channels,
                        out_channels=in_channels,
                        ch_per_head=in_channels // attention_num_heads,
                        num_heads=attention_num_heads,
                        norm_name=("GROUP", {'num_groups': resnet_groups, "affine": True}),
                        dropout=dropout,
                        emb_dim=temb_channels,
                        attention_type='linear'
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    output_scale_factor=output_scale_factor
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states, temb)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states
    

