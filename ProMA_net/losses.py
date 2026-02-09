
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Sequence, Tuple, Optional
import numpy as np
class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)
class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice



class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()





def avg_pool3d_deterministic(x: torch.Tensor, radius: int):
    k = 2 * radius + 1
    C = x.shape[1]
    w = torch.ones((C, 1, k, k, k), device=x.device) / (k**3)
    return F.conv3d(x, w, bias=None, stride=1, padding=radius, groups=C)


def pad_replicate_3d(x, pz, py, px):
    if pz: x = torch.cat([x[:,:, :1].repeat(1,1,pz,1,1), x, x[:,:,-1:].repeat(1,1,pz,1,1)], 2)
    if py: x = torch.cat([x[:,:,:, :1].repeat(1,1,1,py,1), x, x[:,:,:,-1:].repeat(1,1,1,py,1)], 3)
    if px: x = torch.cat([x[:,:,:,:, :1].repeat(1,1,1,1,px), x, x[:,:,:,:,-1:].repeat(1,1,1,1,px)], 4)
    return x

def pdist_squared(x):
    xx = (x ** 2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    return torch.clamp(dist, 0.0, np.inf)

def MINDSSC(img, radius=2, dilation=2):
    # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor
    # kernel size
    kernel_size = radius * 2 + 1

    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.tensor([[0, 1, 1],
                                      [1, 1, 0],
                                      [1, 0, 1],
                                      [1, 1, 2],
                                      [2, 1, 1],
                                      [1, 2, 1]]).long()

    # squared distances
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
    mask = ((x > y).view(-1) & (dist == 2).view(-1))

    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)

    ssd = F.avg_pool3d(rpad2((F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2),
                       kernel_size, stride=1)

    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item())
    mind /= mind_var
    mind = torch.exp(-mind)

    # permute to have same ordering as C++ code
    mind = mind[:, torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

    return mind
def MINDSSC_cpu(img, radius=2, dilation=2):
    orig_device = img.device

    six_neighbourhood = torch.tensor([
        [0, 1, 1], [1, 1, 0], [1, 0, 1],
        [1, 1, 2], [2, 1, 1], [1, 2, 1]
    ]).long()
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6), indexing='ij')
    mask = ((x > y).view(-1) & (dist == 2).view(-1))

    idx1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask]
    idx2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask]
    mshift1 = torch.zeros(12, 1, 3, 3, 3, device='cpu')
    mshift2 = torch.zeros(12, 1, 3, 3, 3, device='cpu')
    mshift1.view(-1)[torch.arange(12) * 27 + idx1[:, 0] * 9 + idx1[:, 1] * 3 + idx1[:, 2]] = 1
    mshift2.view(-1)[torch.arange(12) * 27 + idx2[:, 0] * 9 + idx2[:, 1] * 3 + idx2[:, 2]] = 1
    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)
    img_cpu = img.cpu()
    patch = (F.conv3d(rpad1(img_cpu), mshift1, dilation=dilation)
             - F.conv3d(rpad1(img_cpu), mshift2, dilation=dilation)) ** 2
    # patch = (conv3d_deterministic(rpad1(img), mshift1, dilation=dilation)
    #        - conv3d_deterministic(rpad2(img), mshift2, dilation=dilation)
    #         ) ** 2
    # patch = patch.to(orig_device)
    # rpad2 = rpad2.to(orig_device)
    ssd = F.avg_pool3d(rpad2(patch), kernel_size=2 * radius + 1, stride=1)
    ssd = ssd.to(orig_device)
    mind = ssd - ssd.min(dim=1, keepdim=True)[0]
    var = mind.mean(dim=1, keepdim=True)
    var = torch.clamp(var, var.mean() * 0.001, var.mean() * 1000)
    mind = torch.exp(-mind / var)

    order = torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3])
    mind = mind[:, order, ...]

    return mind

class MultiScaleMINDCosine(nn.Module):
    def __init__(
        self,
        patch_sizes   : Sequence[int] = (3, 5, 9),
        radius_map    : Optional[Sequence[int]] = (3, 2, 1),
        scale_weights : Optional[Sequence[float]] = None,
        learnable     : bool = True,
        chunk_size    : int = 1024,
        temperature   : float = 1,
    ):
        super().__init__()
        assert len(patch_sizes) == len(radius_map)
        self.ps    = list(patch_sizes)
        self.Rmap  = list(radius_map)
        self.chunk = chunk_size

        assert temperature > 0.0, "temperature τ must be > 0"
        self.tau = float(temperature)

        if scale_weights is None:
            scale_weights = [1 / len(self.ps)] * len(self.ps)
        w = torch.tensor(scale_weights, dtype=torch.float32)
        self.scale_w = nn.Parameter(w) if learnable else w

    @staticmethod
    def _unfold(x, p):
        B, C, D, H, W = x.shape
        pz, py, px = (p - D % p) % p, (p - H % p) % p, (p - W % p) % p
        if pz or py or px:
            x = pad_replicate_3d(x, pz, py, px)
            D += pz; H += py; W += px
        Nz, Ny, Nx = D // p, H // p, W // p
        patches = (
            x.view(B, C, Nz, p, Ny, p, Nx, p)
             .permute(0, 2, 4, 6, 1, 3, 5, 7)
             .reshape(B, Nz * Ny * Nx, C * p**3)
        )
        return patches, (Nz, Ny, Nx)

    def _local_loss(self, pa, pb, grid, R):
        # 归一化 → 余弦相似度
        pa = F.normalize(pa, dim=-1)
        pb = F.normalize(pb, dim=-1)

        B, N, _ = pa.shape
        pbT = pb.transpose(1, 2)

        tot = 0.0
        cnt = 0.0

        for s in range(0, N, self.chunk):
            pa_c = pa[:, s:s + self.chunk]
            bc = pa_c.size(1)

            sim = torch.einsum('bnc,bcm->bnm', pa_c, pbT)  # [B, bc, N]

            gi = grid[s:s + bc]
            keep = (
                (gi[:, 0, None] - grid[:, 0]).abs() <= R
            ) & (
                (gi[:, 1, None] - grid[:, 1]).abs() <= R
            ) & (
                (gi[:, 2, None] - grid[:, 2]).abs() <= R
            )
            sim = sim.masked_fill(~keep[None].to(sim.device), -1e9)

            sim = sim / self.tau

            lse = torch.logsumexp(sim, -1)  # [B, bc]

            col = torch.arange(s, s + bc, device=sim.device).clamp_max(pb.size(1) - 1)
            aligned = sim[
                torch.arange(B, device=sim.device)[:, None],
                torch.arange(bc, device=sim.device)[None, :],
                col[None, :]
            ]  # [B, bc]

            loss = -(aligned - lse)  # [B, bc]
            tot += loss.sum()
            cnt += loss.numel()

        return tot / cnt

    def forward(self, fixed, moving):
        mind_f = MINDSSC(fixed)
        mind_m = MINDSSC(moving)

        weights = F.softmax(self.scale_w, 0) if isinstance(self.scale_w, torch.Tensor) else self.scale_w

        total = 0.0
        for p, R, w in zip(self.ps, self.Rmap, weights):
            pf, shape = self._unfold(mind_f, p)
            pm, _     = self._unfold(mind_m, p)

            Nz, Ny, Nx = shape
            device = fixed.device
            gz, gy, gx = torch.meshgrid(
                torch.arange(Nz, device=device),
                torch.arange(Ny, device=device),
                torch.arange(Nx, device=device),
                indexing='ij'
            )
            grid = torch.stack([gz, gy, gx], -1).reshape(-1, 3)

            l = self._local_loss(pf, pm, grid, R)
            total += w * l

        return total
