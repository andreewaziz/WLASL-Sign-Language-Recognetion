from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim: int, size: int) -> int:
        if size % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        return max(self.kernel_size[dim] - (size % self.stride[dim]), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, t, h, w = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_front = pad_t // 2
        pad_h_front = pad_h // 2
        pad_w_front = pad_w // 2
        pad = (
            pad_w_front,
            pad_w - pad_w_front,
            pad_h_front,
            pad_h - pad_h_front,
            pad_t_front,
            pad_t - pad_t_front,
        )
        return super().forward(F.pad(x, pad))


class Unit3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        output_channels: int,
        kernel_shape: tuple[int, int, int] | list[int] = (1, 1, 1),
        stride: tuple[int, int, int] | list[int] = (1, 1, 1),
        padding: int | tuple[int, int, int] = 0,
        activation_fn=F.relu,
        use_batch_norm: bool = True,
        use_bias: bool = False,
        name: str = "unit_3d",
    ) -> None:
        super().__init__()
        self._output_channels = output_channels
        self._kernel_shape = tuple(kernel_shape)
        self._stride = tuple(stride)
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self._output_channels,
            kernel_size=self._kernel_shape,
            stride=self._stride,
            padding=0,
            bias=self._use_bias,
        )

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim: int, size: int) -> int:
        if size % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        return max(self._kernel_shape[dim] - (size % self._stride[dim]), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, t, h, w = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_front = pad_t // 2
        pad_h_front = pad_h // 2
        pad_w_front = pad_w // 2
        pad = (
            pad_w_front,
            pad_w - pad_w_front,
            pad_h_front,
            pad_h - pad_h_front,
            pad_t_front,
            pad_t - pad_t_front,
        )

        x = self.conv3d(F.pad(x, pad))
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: list[int], name: str) -> None:
        super().__init__()
        self.b0 = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_0/Conv3d_0a_1x1",
        )
        self.b1a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_1/Conv3d_0a_1x1",
        )
        self.b1b = Unit3D(
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_shape=[3, 3, 3],
            name=name + "/Branch_1/Conv3d_0b_3x3",
        )
        self.b2a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_2/Conv3d_0a_1x1",
        )
        self.b2b = Unit3D(
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_shape=[3, 3, 3],
            name=name + "/Branch_2/Conv3d_0b_3x3",
        )
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_3/Conv3d_0b_1x1",
        )
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.b0(x), self.b1b(self.b1a(x)), self.b2b(self.b2a(x)), self.b3b(self.b3a(x))],
            dim=1,
        )


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture compatible with the original WLASL weights."""

    VALID_ENDPOINTS = (
        "Conv3d_1a_7x7",
        "MaxPool3d_2a_3x3",
        "Conv3d_2b_1x1",
        "Conv3d_2c_3x3",
        "MaxPool3d_3a_3x3",
        "Mixed_3b",
        "Mixed_3c",
        "MaxPool3d_4a_3x3",
        "Mixed_4b",
        "Mixed_4c",
        "Mixed_4d",
        "Mixed_4e",
        "Mixed_4f",
        "MaxPool3d_5a_2x2",
        "Mixed_5b",
        "Mixed_5c",
        "Logits",
        "Predictions",
    )

    def __init__(
        self,
        num_classes: int = 400,
        spatial_squeeze: bool = True,
        final_endpoint: str = "Logits",
        name: str = "inception_i3d",
        in_channels: int = 3,
        dropout_keep_prob: float = 0.5,
    ) -> None:
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError(f"Unknown final endpoint {final_endpoint}")

        super().__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None
        self.end_points: OrderedDict[str, nn.Module] = OrderedDict()

        self._add_endpoint(
            "Conv3d_1a_7x7",
            Unit3D(in_channels, 64, kernel_shape=[7, 7, 7], stride=(2, 2, 2), padding=(3, 3, 3), name=name + "Conv3d_1a_7x7"),
        )
        self._add_endpoint("MaxPool3d_2a_3x3", MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0))
        self._add_endpoint("Conv3d_2b_1x1", Unit3D(64, 64, kernel_shape=[1, 1, 1], padding=0, name=name + "Conv3d_2b_1x1"))
        self._add_endpoint("Conv3d_2c_3x3", Unit3D(64, 192, kernel_shape=[3, 3, 3], padding=1, name=name + "Conv3d_2c_3x3"))
        self._add_endpoint("MaxPool3d_3a_3x3", MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0))
        self._add_endpoint("Mixed_3b", InceptionModule(192, [64, 96, 128, 16, 32, 32], name + "Mixed_3b"))
        self._add_endpoint("Mixed_3c", InceptionModule(256, [128, 128, 192, 32, 96, 64], name + "Mixed_3c"))
        self._add_endpoint("MaxPool3d_4a_3x3", MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0))
        self._add_endpoint("Mixed_4b", InceptionModule(480, [192, 96, 208, 16, 48, 64], name + "Mixed_4b"))
        self._add_endpoint("Mixed_4c", InceptionModule(512, [160, 112, 224, 24, 64, 64], name + "Mixed_4c"))
        self._add_endpoint("Mixed_4d", InceptionModule(512, [128, 128, 256, 24, 64, 64], name + "Mixed_4d"))
        self._add_endpoint("Mixed_4e", InceptionModule(512, [112, 144, 288, 32, 64, 64], name + "Mixed_4e"))
        self._add_endpoint("Mixed_4f", InceptionModule(528, [256, 160, 320, 32, 128, 128], name + "Mixed_4f"))
        self._add_endpoint("MaxPool3d_5a_2x2", MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0))
        self._add_endpoint("Mixed_5b", InceptionModule(832, [256, 160, 320, 32, 128, 128], name + "Mixed_5b"))
        self._add_endpoint("Mixed_5c", InceptionModule(832, [384, 192, 384, 48, 128, 128], name + "Mixed_5c"))

        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(
            in_channels=1024,
            output_channels=self._num_classes,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name="logits",
        )

    def _add_endpoint(self, name: str, module: nn.Module) -> None:
        if name in self.VALID_ENDPOINTS:
            self.end_points[name] = module
            self.add_module(name, module)

    def replace_logits(self, num_classes: int) -> None:
        self._num_classes = num_classes
        self.logits = Unit3D(
            in_channels=1024,
            output_channels=self._num_classes,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name="logits",
        )

    def forward(self, x: torch.Tensor, pretrained: bool = False, n_tune_layers: int = -1) -> torch.Tensor:
        if pretrained:
            if n_tune_layers < 0:
                raise ValueError("n_tune_layers must be non-negative when pretrained=True")
            freeze_endpoints = self.VALID_ENDPOINTS[:-n_tune_layers] if n_tune_layers else self.VALID_ENDPOINTS
            tune_endpoints = self.VALID_ENDPOINTS[-n_tune_layers:] if n_tune_layers else []
        else:
            freeze_endpoints = []
            tune_endpoints = self.VALID_ENDPOINTS

        with torch.no_grad():
            for end_point in freeze_endpoints:
                if end_point in self.end_points:
                    x = self._modules[end_point](x)

        for end_point in tune_endpoints:
            if end_point in self.end_points:
                x = self._modules[end_point](x)

        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            x = x.squeeze(3).squeeze(3)
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)


def _unwrap_state_dict(raw: Any) -> dict[str, torch.Tensor]:
    if isinstance(raw, dict) and "model" in raw:
        raw = raw["model"]
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    if not isinstance(raw, dict):
        raise TypeError("Checkpoint does not contain a state dict")
    return {k.removeprefix("module."): v for k, v in raw.items()}


def load_kinetics_pretrained(model: InceptionI3d, path: str | Path, device: torch.device) -> None:
    state = _unwrap_state_dict(torch.load(path, map_location=device))
    model.load_state_dict(state, strict=True)


def build_i3d(
    num_classes: int,
    device: torch.device,
    pretrained: str | Path | None = None,
    in_channels: int = 3,
) -> InceptionI3d:
    if pretrained:
        model = InceptionI3d(400, in_channels=in_channels)
        load_kinetics_pretrained(model, pretrained, device=torch.device("cpu"))
        model.replace_logits(num_classes)
    else:
        model = InceptionI3d(num_classes, in_channels=in_channels)
    return model.to(device)
