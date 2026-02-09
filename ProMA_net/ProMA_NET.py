import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_, to_3tuple
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from .modelio import LoadableModel, store_config_args

# Vector Integration and Resize Transform classes (unchanged)
################################################################

from enum import Enum
from torch.nn import functional as nnf
class VecInt(nn.Module):
    def __init__(self, inshape, nsteps):
        super().__init__()
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class ResizeTransform(nn.Module):
    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x
        elif self.factor > 1:
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
        return x

# MLP and Window-related functions (unchanged)
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, L, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], L // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows

def window_reverse(windows, window_size, H, W, L):
    B = int(windows.shape[0] / (H * W * L / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, H // window_size[0], W // window_size[1], L // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, L, -1)
    return x

def chessboard_shuffle(x_a_windows, x_b_windows, window_size, Hp, Wp, Tp):

    grid_depth = Hp // window_size[0]
    grid_height = Wp // window_size[1]
    grid_width = Tp // window_size[2]
    num_windows = grid_depth * grid_height * grid_width

    assert x_a_windows.shape[0] == num_windows, \
        f"Window count mismatch: input has {x_a_windows.shape[0]} windows, but based on the dimensions it should have {num_windows}."

    C = x_a_windows.shape[-1]
    ws0, ws1, ws2 = window_size

    x_a_grid = x_a_windows.view(grid_depth, grid_height, grid_width, ws0, ws1, ws2, C)
    x_b_grid = x_b_windows.view(grid_depth, grid_height, grid_width, ws0, ws1, ws2, C)

    device = x_a_windows.device
    i = torch.arange(grid_depth, device=device)[:, None, None]
    j = torch.arange(grid_height, device=device)[None, :, None]
    k = torch.arange(grid_width, device=device)[None, None, :]
    mask = (i + j + k) % 2 == 0


    mask_expanded = mask.unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)
    mask_expanded = mask_expanded.expand(-1, -1, -1, ws0, ws1, ws2, C)


    x_a_shuffled = torch.where(mask_expanded, x_a_grid, x_b_grid)
    x_b_shuffled = torch.where(mask_expanded, x_b_grid, x_a_grid)

    return (
        x_a_shuffled.reshape(num_windows, ws0, ws1, ws2, C),
        x_b_shuffled.reshape(num_windows, ws0, ws1, ws2, C))


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, rpe=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        # dim = dim // 2
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t]))
        coords_flatten = torch.flatten(coords, 1)
        self.rpe = rpe
        if self.rpe:
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GateMLP(nn.Module):

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, 1, bias=True)
        )
        nn.init.constant_(self.fc[-1].bias, -2.0)

    def forward(self, x):
        g = self.fc(x.mean(1, keepdim=False))
        g = torch.sigmoid(g)
        return g.view(-1, 1, 1)

class SwinTransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(7, 7, 7),
                 shift_size=(0, 0, 0),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 rpe=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= min(shift_size) < min(window_size), "shift_size must in 0-window_size"

        c_half = dim // 2

        self.norm1_a = norm_layer(c_half)
        self.norm1_b = norm_layer(c_half)

        from timm.models.layers import DropPath, trunc_normal_
        self.attn_a = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, rpe=rpe,
            attn_drop=attn_drop, proj_drop=drop)
        self.attn_b = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, rpe=rpe,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path_a = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_b = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2_a = norm_layer(c_half)
        self.norm2_b = norm_layer(c_half)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_a = Mlp(in_features=c_half, hidden_features=mlp_hidden_dim,
                         act_layer=act_layer, drop=drop)
        self.mlp_b = Mlp(in_features=c_half, hidden_features=mlp_hidden_dim,
                         act_layer=act_layer, drop=drop)

        self.gate_mlp = GateMLP(c_half)
        # forward 时会赋值
        self.H = self.W = self.T = None


    def forward(self, x_a, x_b, mask_matrix):

        H, W, T = self.H, self.W, self.T
        B, L, C_half = x_a.shape
        C = C_half
        assert L == H * W * T, "input feature has wrong size"
        ws0, ws1, ws2 = self.window_size
        if min(self.shift_size) > 0:
            shift_window = True
        else:
            shift_window = False
        shortcut_a, shortcut_b = x_a, x_b
        x_a = self.norm1_a(x_a).view(B, H, W, T, C_half)
        x_b = self.norm1_b(x_b).view(B, H, W, T, C_half)

        pad_l = pad_t = pad_f = 0
        pad_r = (ws0 - H % ws0) % ws0
        pad_b = (ws1 - W % ws1) % ws1
        pad_h = (ws2 - T % ws2) % ws2
        if pad_r or pad_b or pad_h:
            x_a = nnf.pad(x_a, (0, 0, pad_f, pad_h, pad_t, pad_b, pad_l, pad_r))
            x_b = nnf.pad(x_b, (0, 0, pad_f, pad_h, pad_t, pad_b, pad_l, pad_r))
        _, Hp, Wp, Tp, _ = x_a.shape

        if shift_window:
            x_a_nd_ws = window_partition(x_a, self.window_size)
            x_b_nd_ws = window_partition(x_b, self.window_size)
            x_a_cross, x_b_cross = chessboard_shuffle(
                x_a_nd_ws, x_b_nd_ws, self.window_size, Hp, Wp, Tp)
            x_a_cross = window_reverse(x_a_cross, self.window_size, Hp, Wp, Tp)
            x_b_cross = window_reverse(x_b_cross, self.window_size, Hp, Wp, Tp)

            if min(self.shift_size) > 0:
                shifts = (-self.shift_size[0], -self.shift_size[1], -self.shift_size[2])
                shifted_x_a = torch.roll(x_a, shifts=shifts, dims=(1, 2, 3))
                shifted_x_b = torch.roll(x_b, shifts=shifts, dims=(1, 2, 3))
                shifted_x_a_cross = torch.roll(x_a_cross, shifts=shifts, dims=(1, 2, 3))
                shifted_x_b_cross = torch.roll(x_b_cross, shifts=shifts, dims=(1, 2, 3))
                attn_mask = mask_matrix
            else:
                shifted_x_a, shifted_x_b = x_a, x_b
                shifted_x_a_cross, shifted_x_b_cross = x_a_cross, x_b_cross
                attn_mask = None

            x_a_ws = window_partition(shifted_x_a, self.window_size)
            x_b_ws = window_partition(shifted_x_b, self.window_size)
            x_a_ws_cross = window_partition(shifted_x_a_cross, self.window_size)
            x_b_ws_cross = window_partition(shifted_x_b_cross, self.window_size)

            nW_B, _, _, _, C_half = x_a_ws.shape
            Ntokens = ws0 * ws1 * ws2
            x_a_ws = x_a_ws.view(-1, Ntokens, C_half)
            x_b_ws = x_b_ws.view(-1, Ntokens, C_half)
            x_a_ws_cross = x_a_ws_cross.view(-1, Ntokens, C_half)
            x_b_ws_cross = x_b_ws_cross.view(-1, Ntokens, C_half)

            x_ws = torch.cat([x_a_ws, x_b_ws], dim=2)
            x_ws_cross = torch.cat([x_a_ws_cross, x_b_ws_cross], dim=2)

            attn_out_a = self.attn_a(x_ws, mask=attn_mask)
            attn_out_b = self.attn_b(x_ws_cross, mask=attn_mask)


            seq1_restored = attn_out_a[..., :C_half]
            seq2_restored = attn_out_a[..., C_half:]
            seq1_cross_restored = attn_out_b[..., :C_half]
            seq2_cross_restored = attn_out_b[..., C_half:]

            a_ws = seq1_restored.view(-1, ws0, ws1, ws2, C_half)
            b_ws = seq2_restored.view(-1, ws0, ws1, ws2, C_half)
            a_ws_cross = seq1_cross_restored.view(-1, ws0, ws1, ws2, C_half)
            b_ws_cross = seq2_cross_restored.view(-1, ws0, ws1, ws2, C_half)

            # 逆窗口操作
            shifted_x_a = window_reverse(a_ws, self.window_size, Hp, Wp, Tp)
            shifted_x_b = window_reverse(b_ws, self.window_size, Hp, Wp, Tp)
            shifted_x_a_cross = window_reverse(a_ws_cross, self.window_size, Hp, Wp, Tp)
            shifted_x_b_cross = window_reverse(b_ws_cross, self.window_size, Hp, Wp, Tp)

            if min(self.shift_size) > 0:
                x_a = torch.roll(shifted_x_a, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                                 dims=(1, 2, 3))
                x_b = torch.roll(shifted_x_b, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                                 dims=(1, 2, 3))
                x_a_cross = torch.roll(shifted_x_a_cross,
                                       shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                                       dims=(1, 2, 3))
                x_b_cross = torch.roll(shifted_x_b_cross,
                                       shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                                       dims=(1, 2, 3))
            else:
                x_a = shifted_x_a
                x_b = shifted_x_b
                x_a_cross = shifted_x_a_cross
                x_b_cross = shifted_x_b_cross

            x_a_windows_cross = window_partition(x_a_cross, self.window_size)
            x_b_windows_cross = window_partition(x_b_cross, self.window_size)
            x_a_cross, x_b_cross = chessboard_shuffle(x_a_windows_cross, x_b_windows_cross, self.window_size, Hp, Wp,
                                                      Tp)
            x_a = window_partition(x_a, self.window_size)
            x_b = window_partition(x_b, self.window_size)
            gate_a = self.gate_mlp(x_a_cross.view(-1, Ntokens, C_half))
            gate_b = self.gate_mlp(x_b_cross.view(-1, Ntokens, C_half))

            gate_a_tok = gate_a.expand(-1, Ntokens, 1).view(-1, ws0, ws1, ws2, 1)
            gate_b_tok = gate_b.expand(-1, Ntokens, 1).view(-1, ws0, ws1, ws2, 1)

            x_a = x_a + gate_a_tok * x_a_cross
            x_b = x_b + gate_b_tok * x_b_cross

            x_a = window_reverse(x_a, self.window_size, Hp, Wp, Tp)
            x_b = window_reverse(x_b, self.window_size, Hp, Wp, Tp)
            if pad_r > 0 or pad_b > 0 or pad_h > 0:
                x_a = x_a[:, :H, :W, :T, :].contiguous()
                x_b = x_b[:, :H, :W, :T, :].contiguous()

            x_a = x_a.view(B, H * W * T, C)
            x_b = x_b.view(B, H * W * T, C)
            x_a = shortcut_a + self.drop_path_a(x_a)
            x_b = shortcut_b + self.drop_path_b(x_b)

            # FFN
            x_a = x_a + self.drop_path_a(self.mlp_a(self.norm2_a(x_a)))
            x_b = x_b + self.drop_path_b(self.mlp_b(self.norm2_b(x_b)))

            return x_a, x_b

        else:
            if min(self.shift_size) > 0:
                shifts = (-self.shift_size[0], -self.shift_size[1], -self.shift_size[2])
                shifted_x_a = torch.roll(x_a, shifts=shifts, dims=(1, 2, 3))
                shifted_x_b = torch.roll(x_b, shifts=shifts, dims=(1, 2, 3))
                attn_mask = mask_matrix
            else:
                shifted_x_a, shifted_x_b = x_a, x_b
                attn_mask = None

            x_a_ws = window_partition(shifted_x_a, self.window_size)
            x_b_ws = window_partition(shifted_x_b, self.window_size)

            nW_B, _, _, _, C_half = x_a_ws.shape
            Ntokens = ws0 * ws1 * ws2
            x_a_ws = x_a_ws.view(-1, Ntokens, C_half)
            x_b_ws = x_b_ws.view(-1, Ntokens, C_half)

            x_ws = torch.cat([x_a_ws, x_b_ws], dim=2)
            attn_out = self.attn_a(x_ws, mask=attn_mask)

            x_restored = attn_out
            seq1_restored = x_restored[..., :C_half]
            seq2_restored = x_restored[..., C_half:]

            a_ws = seq1_restored.view(-1, ws0, ws1, ws2, C_half)
            b_ws = seq2_restored.view(-1, ws0, ws1, ws2, C_half)

            shifted_x_a = window_reverse(a_ws, self.window_size, Hp, Wp, Tp)
            shifted_x_b = window_reverse(b_ws, self.window_size, Hp, Wp, Tp)

            if min(self.shift_size) > 0:
                x_a = torch.roll(shifted_x_a, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                                 dims=(1, 2, 3))
                x_b = torch.roll(shifted_x_b, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                                 dims=(1, 2, 3))
            else:
                x_a = shifted_x_a
                x_b = shifted_x_b

            x_a = window_partition(x_a, self.window_size)
            x_b = window_partition(x_b, self.window_size)

            x_a = window_reverse(x_a, self.window_size, Hp, Wp, Tp)
            x_b = window_reverse(x_b, self.window_size, Hp, Wp, Tp)
            if pad_r > 0 or pad_b > 0 or pad_h > 0:
                x_a = x_a[:, :H, :W, :T, :].contiguous()
                x_b = x_b[:, :H, :W, :T, :].contiguous()

            x_a = x_a.view(B, H * W * T, C)
            x_b = x_b.view(B, H * W * T, C)
            # 残差
            x_a = shortcut_a + self.drop_path_a(x_a)
            x_b = shortcut_b + self.drop_path_b(x_b)

            # FFN
            x_a = x_a + self.drop_path_a(self.mlp_a(self.norm2_a(x_a)))
            x_b = x_b + self.drop_path_b(self.mlp_b(self.norm2_b(x_b)))

            return x_a, x_b

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        # dim = dim // 2
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, (8//reduce_factor) * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x, H, W, T):
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0 and T % 2 == 0, f"x size ({H}*{W}*{T}) are not even."
        x = x.view(B, H, W, T, C)
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x = nnf.pad(x, (0, 0, 0, T % 2, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 0::2, 1::2, 1::2, :]
        x6 = x[:, 1::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = x.view(B, -1, 8 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 rpe=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 pat_merg_rf=2):
        super().__init__()
        self.window_size = window_size
        self.shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.pat_merg_rf = pat_merg_rf
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                rpe=rpe,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])
        self.downsample = downsample(dim=dim // 2, norm_layer=norm_layer, reduce_factor=pat_merg_rf) if downsample is not None else None

    def forward(self, x_a, x_b, H, W, T):
        B, N, C = x_a.shape
        c = C // 2
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        Tp = int(np.ceil(T / self.window_size[2])) * self.window_size[2]
        img_mask = torch.zeros((1, Hp, Wp, Tp, 1), device=x_a.device)
        h_slices = (slice(0, -self.window_size[0]), slice(-self.window_size[0], -self.shift_size[0]), slice(-self.shift_size[0], None))
        w_slices = (slice(0, -self.window_size[1]), slice(-self.window_size[1], -self.shift_size[1]), slice(-self.shift_size[1], None))
        t_slices = (slice(0, -self.window_size[2]), slice(-self.window_size[2], -self.shift_size[2]), slice(-self.shift_size[2], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                for t in t_slices:
                    img_mask[:, h, w, t, :] = cnt
                    cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W, blk.T = H, W, T
            if self.use_checkpoint:
                x_a, x_b = checkpoint.checkpoint(blk, x_a, x_b, attn_mask)
            else:
                x_a, x_b = blk(x_a, x_b, attn_mask)

        if self.downsample is not None:
            x_a_down = self.downsample(x_a, H, W, T)
            x_b_down = self.downsample(x_b, H, W, T)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x_a, x_b, H, W, T, x_a_down, x_b_down, Wh, Ww, Wt
        else:
            return x_a, x_b, H, W, T, x_a, x_b, H, W, T

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        _, _, H, W, T = x.size()
        if T % self.patch_size[2] != 0:
            x = nnf.pad(x, (0, self.patch_size[2] - T % self.patch_size[2]))
        if W % self.patch_size[1] != 0:
            x = nnf.pad(x, (0, 0, 0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = nnf.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)
        return x


class SinPositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        channels = int(np.ceil(channels/6)*2)
        if channels % 2:
            channels += 1
        self.channels = channels
        self.inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x,y,z,self.channels*3),device=tensor.device).type(tensor.type())
        emb[:,:,:,:self.channels] = emb_x
        emb[:,:,:,self.channels:2*self.channels] = emb_y
        emb[:,:,:,2*self.channels:] = emb_z
        emb = emb[None,:,:,:,:orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return emb.permute(0, 4, 1, 2, 3)


class SwinTransformer(nn.Module):
    def __init__(self, pretrain_img_size=224,
                 patch_size=4,
                 in_chans=1,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 rpe=True,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=4):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # Two separate PatchEmbeds
        self.patch_embed_a = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim//2, norm_layer=norm_layer if patch_norm else None)
        self.patch_embed_b = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim//2, norm_layer=norm_layer if patch_norm else None)

        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1], pretrain_img_size[2] // patch_size[2]]
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinPositionalEncoding3D(embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                rpe=rpe,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                pat_merg_rf=pat_merg_rf)
            self.layers.append(layer)

        self.num_features = [int(embed_dim * (2 ** i)) for i in range(self.num_layers)]
        for i_layer in out_indices:
            layer = norm_layer(self.num_features[i_layer] // 2)
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def forward(self, source, target):
        x_a = self.patch_embed_a(source)
        x_b = self.patch_embed_b(target)

        Wh, Ww, Wt = x_a.size(2), x_a.size(3), x_a.size(4)
        if self.ape:
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww, Wt), mode='trilinear')
            x_a = (x_a + absolute_pos_embed).flatten(2).transpose(1, 2)
            x_b = (x_b + absolute_pos_embed).flatten(2).transpose(1, 2)
        elif self.spe:
            x_a = (x_a + self.pos_embd(x_a)).flatten(2).transpose(1, 2)
            x_b = (x_b + self.pos_embd(x_b)).flatten(2).transpose(1, 2)
        else:
            x_a = x_a.flatten(2).transpose(1, 2)
            x_b = x_b.flatten(2).transpose(1, 2)
        x_a = self.pos_drop(x_a)
        x_b = self.pos_drop(x_b)
        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_a_out, x_b_out, H, W, T, x_a, x_b, Wh, Ww, Wt = layer(x_a, x_b, Wh, Ww, Wt)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_a_out)
                out = x_out.view(-1, H, W, T, self.num_features[i]//2).permute(0, 4, 1, 2, 3).contiguous()
                outs.append(out)
        return outs

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed_a.eval()
            self.patch_embed_b.eval()
            for param in self.patch_embed_a.parameters():
                param.requires_grad = False
            for param in self.patch_embed_b.parameters():
                param.requires_grad = False
        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False
        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')


class Conv3dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        relu = nn.LeakyReLU(inplace=True)
        nm = nn.InstanceNorm3d(out_channels) if not use_batchnorm else nn.BatchNorm3d(out_channels)
        super().__init__(conv, nm, relu)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv3dReLU(in_channels + skip_channels + skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2 = Conv3dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv3 = Conv3dReLU(out_channels, out_channels,kernel_size=3,padding=1,use_batchnorm=use_batchnorm)
        self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
    def forward(self, x, skip=None, skip2=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        if skip2 is not None:
            x = torch.cat([x, skip2], dim=1)
            x = self.conv1(x)
        if skip2 is None:
            x = self.conv2(x)
        x = self.conv3(x)
        return x

class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)

class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class ProMANET(LoadableModel):
    @store_config_args
    def __init__(self, config):
        super().__init__()

        self.if_convskip = config.if_convskip
        self.if_transskip = config.if_transskip
        embed_dim = config.embed_dim
        inshape = config.img_size

        self.transformer = SwinTransformer(
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=embed_dim,
            depths=config.depths,
            num_heads=config.num_heads,
            window_size=config.window_size,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            ape=config.ape,
            spe=config.spe,
            rpe=config.rpe,
            patch_norm=config.patch_norm,
            use_checkpoint=config.use_checkpoint,
            out_indices=config.out_indices,
            pat_merg_rf=config.pat_merg_rf)


        self.up0 = DecoderBlock(embed_dim * 4, embed_dim * 2, skip_channels=embed_dim * 2 if self.if_transskip else 0,use_batchnorm=False)
        self.up1 = DecoderBlock(embed_dim * 2, embed_dim, skip_channels=embed_dim if self.if_transskip else 0,use_batchnorm=False)
        self.up2 = DecoderBlock(embed_dim , embed_dim //2, skip_channels=embed_dim//2 if self.if_transskip else 0,use_batchnorm=False)
        self.up3 = DecoderBlock(embed_dim//2, embed_dim // 4, skip_channels=embed_dim // 4 if self.if_convskip else 0,use_batchnorm=False)
        self.up4 = DecoderBlock(embed_dim // 4, config.reg_head_chan//2,skip_channels=config.reg_head_chan//2 if self.if_convskip else 0, use_batchnorm=False)

        self.c1 = Conv3dReLU(2, embed_dim//4, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(2, config.reg_head_chan//2, 3, 1, use_batchnorm=False)
        self.c3 = Conv3dReLU(embed_dim * 8, embed_dim * 4, 3, 1, use_batchnorm=False)


        self.spatial_trans = SpatialTransformer(config.img_size)
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

        ndims = len(inshape)
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(config.reg_head_chan//2, ndims, kernel_size=3, padding=1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def forward(self, source, target):
        if self.if_convskip:
            xa_s0 = torch.cat([source, target], dim=1)
            xb_s0 = torch.cat([target, source], dim=1)
            xa_s1 = self.avg_pool(xa_s0)

            f4a = self.c1(xa_s1)
            f5a = self.c2(xa_s0)

            xb_s1 = self.avg_pool(xb_s0)

            f4b = self.c1(xb_s1)
            f5b = self.c2(xb_s0)
        else:
            f4a = None
            f5a = None
            f4b = None
            f5b = None


        out_a_feats = self.transformer(source, target)
        out_b_feats = self.transformer(target, source)
        if self.if_transskip:
            f1a = out_a_feats[-2]
            f2a = out_a_feats[-3]
            f3a = out_a_feats[-4]
            f1b = out_b_feats[-2]
            f2b = out_b_feats[-3]
            f3b = out_b_feats[-4]
        else:
            f1a = None
            f2a = None
            f3a = None

        x = torch.cat([out_a_feats[-1], out_b_feats[-1]], dim=1)
        x = self.c3(x)
        x = self.up0(x, f1a, f1b)
        x = self.up1(x, f2a, f2b)
        x = self.up2(x, f3a, f3b)
        x = self.up3(x, f4a, f4b)
        x = self.up4(x, f5a, f5b)
        flow_field = self.flow(x)
        pos_flow = flow_field
        y_source = self.spatial_trans(source, pos_flow)

        return y_source, pos_flow