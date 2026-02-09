import torch
import torch.nn as nn
import torch.nn.functional as nnf


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        # print(self.grid.shape)
        # print(f"flow.shape{flow.shape}")
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

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
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x


##########################################localnet#######################################################
class Activation(nn.Module):
    def __init__(self, identifier: str = "relu"):
        """
        Activation layer that wraps the PyTorch activations.

        :param identifier: activation function name, e.g., "relu", "sigmoid", "tanh"
        """
        super(Activation, self).__init__()

        # Dynamically select the activation function
        if identifier == "relu":
            self.activation = nn.ReLU()
        elif identifier == "sigmoid":
            self.activation = nn.Sigmoid()
        elif identifier == "tanh":
            self.activation = nn.Tanh()
        elif identifier == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif identifier == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation function: {identifier}")

    def forward(self, x):
        return self.activation(x)


class Norm(nn.Module):
    def __init__(self, name: str = "batch_norm", **kwargs):
        """
        Class merges batch norm and layer norm in PyTorch (3D).

        :param name: str, either 'batch_norm' or 'layer_norm'.
        :param axis: int, the axis to normalize, typically for batch normalization.
        :param kwargs: additional arguments to pass to the normalization layer.
        """
        super(Norm, self).__init__()

        if name == "batch_norm":
            # For batch normalization, axis=-1 typically refers to the channel axis (C)
            self._norm = nn.BatchNorm3d(**kwargs)
        elif name == "layer_norm":
            # For layer normalization, we need to specify the normalized shape
            self._norm = nn.LayerNorm(**kwargs)
        else:
            raise ValueError("Unknown normalization type")

    def forward(self, x):
        return self._norm(x)


class MaxPool3d(nn.Module):
    def __init__(
        self,
        pool_size: tuple,
        strides: tuple = None,
        padding: str = "same",
        **kwargs,
    ):
        """
        Layer wraps torch.nn.MaxPool3d.

        :param pool_size: tuple of 3 ints, the size of the pooling window
        :param strides: tuple of 3 ints, the stride of the pooling operation, defaults to pool_size if None
        :param padding: str, 'same' or 'valid', padding type
        :param kwargs: additional arguments.
        """
        super().__init__()

        self.pool_size = pool_size
        self.strides = strides if strides is not None else pool_size
        self.padding = padding

        if self.padding == "same":
            # PyTorch doesn't directly support "same" padding like TensorFlow,
            # so we need to calculate padding manually
            self.padding_size = tuple(
                (pool_size[i] - 1) // 2 for i in range(3)
            )
        else:
            self.padding_size = (0, 0, 0)

    def forward(self, x):
        return nnf.max_pool3d(
            x,
            kernel_size=self.pool_size,
            stride=self.strides,
            padding=self.padding_size
        )


class Conv3d(nn.Module):
    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        strides: int = 1,
        padding: str = "same",
        activation: str = None,
        use_bias: bool = True,
        kernel_initializer: str = "xavier_uniform",
        **kwargs,
    ):
        """
        Layer wraps torch.nn.Conv3d.

        :param filters: number of channels of the output
        :param kernel_size: int or tuple of 3 ints, the size of the convolution kernel
        :param strides: int or tuple of 3 ints, the stride of the convolution
        :param padding: 'same' or 'valid'
        :param activation: activation function name, if None no activation is applied
        :param use_bias: whether to add a bias term
        :param kernel_initializer: initialization method for the weights
        :param kwargs: additional arguments.
        """
        super().__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias

        if self.padding == "same":
            # PyTorch doesn't support 'same' padding directly, so we calculate it manually
            self.padding_size = tuple(
                (kernel_size - 1) // 2 for _ in range(3)
            )
        else:
            self.padding_size = (0, 0, 0)

        # Initialize Conv3d layer
        self.conv3d = nn.Conv3d(
            in_channels=1,  # The input channels should be passed when you instantiate this class
            out_channels=self.filters,
            kernel_size=self.kernel_size,
            stride=self.strides,
            padding=self.padding_size,
            bias=self.use_bias
        )

        # Kernel initializer
        if kernel_initializer == "xavier_uniform":
            nn.init.xavier_uniform_(self.conv3d.weight)

    def forward(self, x):
        return self.conv3d(x)

class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding="same", **kwargs):
        super().__init__()
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,  # The input channels should be passed when you instantiate this class
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strides,
            padding=self._calculate_padding(padding),
            bias=False
        )
        self.norm = Norm()
        self.act = Activation()

    def _calculate_padding(self, padding):
        if padding == "same":
            return tuple(k // 2 for k in self.kernel_size)
        return (0, 0, 0)

    def forward(self, x):
        x = self.conv3d(x)
        x = self.norm(x)
        x = self.act(x)
        return x

