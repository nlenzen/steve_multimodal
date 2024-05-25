"""
From MineCLIP implementation
https://github.com/MineDojo/MineCLIP/blob/main/mineclip/mineclip/pos_embed.py
"""

# Imports
import torch
from einops import rearrange


def interpolate_resize_pos_embed(pos_embed, old_size, new_size):
    """
    NOTE: remove cls token from pos_embed first before passing it here

    Args:
        pos_embed: [seq_len, embed_dim]
        old_size: [h, w], seq_len of pos_embed must be equal to h * w
        new_size: [new_h, new_w]
    """
    old_hw, D = pos_embed.size()
    if isinstance(old_size, int):
        old_size = (old_size, old_size)
    if isinstance(new_size, int):
        new_size = (new_size, new_size)
    assert len(old_size) == 2
    assert len(new_size) == 2
    old_h, old_w = old_size
    assert old_h * old_w == old_hw
    pos_embed = rearrange(pos_embed, "(H W) D -> 1 D H W", H=old_h)
    new_embed = torch.nn.functional.interpolate(
        pos_embed, size=new_size, mode="bicubic", align_corners=False
    )
    new_embed = rearrange(new_embed, "1 D H W -> (H W) D")
    assert new_embed.size() == (new_size[0] * new_size[1], D)
    return new_embed