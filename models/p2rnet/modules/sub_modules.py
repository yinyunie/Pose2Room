#  Copyright (c) 7.2021. Yinyu Nie
#  License: MIT
from torch import nn


def conv(in_channels, out_channels, kernel_size, bias, padding, ndim):
    if ndim == 1:
        return nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
    elif ndim == 2:
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
    elif ndim == 3:
        return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
    else:
        raise NotImplementedError('Unknown ndim.')

def batchnorm(channels, ndim):
    if ndim == 1:
        return nn.BatchNorm1d(channels)
    elif ndim == 2:
        return nn.BatchNorm2d(channels)
    elif ndim == 3:
        return nn.BatchNorm3d(channels)
    else:
        raise NotImplementedError('Unknown ndim.')


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding, ndim, negative_slope=1e-2):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        ndim (1, 2 or 3): n_dim convolution
        negative_slope: for leaky relu if needed
    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(inplace=True, negative_slope=negative_slope)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', conv(in_channels, out_channels, kernel_size, bias, padding=padding, ndim=ndim)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', batchnorm(in_channels, ndim)))
            else:
                modules.append(('batchnorm', batchnorm(out_channels, ndim)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv1/2/3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple):
        ndim (1,2,3): for 1/2/3-D dimensional convolution
        negative_slope: for leaky relu if needed.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8, padding=1, ndim=3,
                 negative_slope=1e-2):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding,
                                        ndim=ndim, negative_slope=negative_slope):
            self.add_module(name, module)
