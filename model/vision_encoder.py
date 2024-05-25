# Imports
import torch
import torch.nn as nn
from .attention import *
from .utils import interpolate_resize_pos_embed

# Vision Transformer ViTb-16 ===========================================================================================

# From https://github.com/MineDojo/MineCLIP/blob/e6c06a0245fac63dceb38bc9bd4fecd033dae735/mineclip/mineclip/base.py
# calculated from 21K video clips, which contains 2.8M frames
MC_IMAGE_MEAN = (0.3331, 0.3245, 0.3051)
MC_IMAGE_STD = (0.2439, 0.2493, 0.2873)


def img_transform(images):
    images = normalize(images / 255.0, mean=MC_IMAGE_MEAN, std=MC_IMAGE_STD)

    return images

def basic_transform(frames):
    assert frames.dim() >= 4
    original_shape = list(frames.size())
    frames = frames.float()
    frames = frames.flatten(0, frames.dim() - 4)
    assert frames.dim() == 4

    B, C, H, W = frames.size()
    assert C % 3 == 0, "channel must divide 3"
    frames = frames.view(B * C // 3, 3, H, W)
    frames = normalize(frames / 255.0, mean=MC_IMAGE_MEAN, std=MC_IMAGE_STD)
    original_shape[-2:] = H, W
    return frames.view(original_shape)


def normalize(tensor: torch.Tensor, mean, std, inplace=False):
    """
    Adapted from https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#normalize

    Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("tensor should be a torch tensor. Got {}.".format(type(tensor)))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(
            f"std evaluated to zero after conversion to {dtype}, leading to division by zero."
        )
    if mean.ndim == 1:
        mean = mean[:, None, None]
    if std.ndim == 1:
        std = std[:, None, None]
    tensor.sub_(mean).div_(std)
    return tensor


class VisionTransformer(nn.Module):
    def __init__(self,
                 resolution,
                 patch_size,
                 width,
                 layers,
                 heads,
                 output_dim):
        super().__init__()

        self.res = resolution
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width**-0.5
        self.cls_token = nn.Parameter(scale * torch.randn(width))
        self.pos_embed = nn.Parameter(scale * torch.randn((resolution // patch_size)**2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)

        self.blocks = nn.Sequential(*[SelfAttentionBlock(width, heads) for _ in range(layers)])

        self.ln_post = nn.LayerNorm(width)
        self.projection = nn.Parameter(scale * torch.randn(width, output_dim))

    def resize_pos_embed(self, new_resolution):
        """
        NOTE: call this method AFTER you load pretrained weights!
        otherwise the weight initialization will fail.
        """
        if isinstance(new_resolution, int):
            new_resolution = (new_resolution, new_resolution)
        else:
            assert len(new_resolution) == 2
        for r in new_resolution:
            assert (
                    r % self.patch_size == 0
            ), f"{new_resolution} is not divisible by {self._patch_size}"

        with torch.no_grad():
            old_embed = self.pos_embed.data.detach()
            cls_embed, old_embed = old_embed[:1], old_embed[1:]
            new_embed = interpolate_resize_pos_embed(
                old_embed,
                self.res // self.patch_size,
                [r // self.patch_size for r in new_resolution],
            )
            self.pos_embed = nn.Parameter(torch.cat([cls_embed, new_embed], dim=0))

    def forward(self, x):
        x = self.conv1(x)   # shape = [*, width, grid, grid]
        b = x.size(0)
        x = x.reshape(b, x.shape[1], -1)    # shape = [*, width, grid**2]
        x = x.permute(0, 2, 1)      # shape = [*, grid**2, width]
        x = torch.cat([self.cls_token.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.pos_embed.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD => LND
        x = self.blocks(x)
        x = x.permute(1, 0, 2)   # LND => NLD

        x = self.ln_post(x[:, 0, :])    # shape = [BatchSize, 1, width]
        if self.projection is not None:
            x = x @ self.projection

        return x
