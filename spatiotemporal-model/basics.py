from typing import Any
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import math
import torch.utils.data
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torch.autograd import Function


class Condition(nn.Module):
    '''
    For Conditional Convolution layer
    '''

    def __init__(self, lmbda_len, channel):
        super(Condition, self).__init__()
        self.s = nn.Linear(lmbda_len, channel)
        self.b = nn.Linear(lmbda_len, channel)
        self.softplus = nn.Softplus()

    def forward(self, x: Tensor, lmbda) -> Tensor:
        # _lambda is one-hot vector like : [0, 1, 0, 0, 0, 0, 0, 0, 0] with 9 lambda points
        weight = self.softplus(self.s(lmbda)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        bias = self.b(lmbda).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        x = x * weight + bias
        return x


class ConditionalConv2d(nn.Module):
    '''
    Conditional Convolution layer
    '''

    def __init__(self, *args: Any, **kwargs: Any):
        super(ConditionalConv2d, self).__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        torch.nn.init.xavier_normal_(self.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.bias.data, 0.01)

        self.ss = nn.Parameter(torch.ones(args[1]).view(1, -1, 1, 1))
        self.bb = nn.Parameter(torch.zeros(args[1]).view(1, -1, 1, 1))
        # self.s = nn.Linear(5, args[1])
        # self.b = nn.Linear(5, args[1])
        # nn.init.constant_(self.s.weight.data, math.log(math.exp(1.0) - 1))
        # nn.init.constant_(self.b.weight.data, math.log(math.exp(1.0) - 1))
        # nn.init.constant_(self.s.bias.data, 0.)
        # nn.init.constant_(self.b.bias.data, 0.)

        self.softplus = nn.Softplus()

    def forward(self, x: Tensor) -> Tensor:
        # _lambda is one-hot vector like : [0, 1, 0, 0, 0, 0, 0, 0, 0] with 9 lambda points
        x = self.conv(x)
        # weight = self.softplus(self.s(lmbda)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        # bias = self.b(lmbda).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        # x = x * weight + bias
        x = x * self.ss + self.bb
        return x


class ConditionalDeconv2d(nn.Module):
    '''
    Conditional Deconvolution layer
    '''

    def __init__(self, *args: Any, **kwargs: Any):
        super(ConditionalDeconv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(*args, **kwargs)
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        torch.nn.init.xavier_normal_(self.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.bias.data, 0.01)

        self.ss = nn.Parameter(torch.ones(args[1]).view(1, -1, 1, 1))
        self.bb = nn.Parameter(torch.zeros(args[1]).view(1, -1, 1, 1))

        # self.s = nn.Linear(5, args[1])
        # self.b = nn.Linear(5, args[1])
        # nn.init.constant_(self.s.weight.data, math.log(math.exp(1.0) - 1))
        # nn.init.constant_(self.b.weight.data, math.log(math.exp(1.0) - 1))
        # nn.init.constant_(self.s.bias.data, 0.)
        # nn.init.constant_(self.b.bias.data, 0.)

        self.softplus = nn.Softplus()

    def forward(self, x: Tensor) -> Tensor:
        # lmbda is one-hot vector like : [0, 1, 0, 0, 0, 0, 0, 0, 0] with 9 lambda points
        x = self.conv(x)
        # weight = self.softplus(self.s(lmbda)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        # bias = self.b(lmbda).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        # x = x * weight + bias
        x = x * self.ss + self.bb
        return x


class MaskedConv2d(nn.Module):
    r"""Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.

    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.

    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super(MaskedConv2d, self).__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        self.weight = self.conv.weight
        self.bias = self.conv.bias

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B"):] = 0
        self.mask[:, :, h // 2 + 1:] = 0

    def forward(self, x: Tensor) -> Tensor:
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return self.conv(x)


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """

    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=0.1,
                 reparam_offset=2 ** -18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset ** 2
        self.beta_bound = ((self.beta_min + self.reparam_offset ** 2) ** 0.5)
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.pedestal = self.pedestal

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta ** 2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma ** 2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs ** 2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


class EntropyParameters(nn.Module):
    '''
    Analysis prior net
    '''

    def __init__(self, N=192, M=320, one_lmbda=False):
        super(EntropyParameters, self).__init__()

        if one_lmbda:
            self.conv1 = nn.Conv2d(N, M // 2, 1, stride=1, padding=0)
            self.conv2 = nn.Conv2d(M // 2, (M * 2) // 3, 1, stride=1, padding=0)
            self.conv3 = nn.Conv2d((M * 2) // 3, M, 1, stride=1, padding=0)
        else:
            self.conv1 = ConditionalConv2d(N, M // 2, 1, stride=1, padding=0)
            self.conv2 = ConditionalConv2d(M // 2, (M * 2) // 3, 1, stride=1, padding=0)
            self.conv3 = ConditionalConv2d((M * 2) // 3, M, 1, stride=1, padding=0)

        self.relu1 = nn.LeakyReLU(inplace=True)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x, lmbda=None):
        if lmbda == None:
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            return self.conv3(x)
        else:
            x = self.relu1(self.conv1(x, lmbda))
            x = self.relu2(self.conv2(x, lmbda))
            return self.conv3(x, lmbda)


class Analysis_prior_net(nn.Module):
    '''
    Analysis prior net
    '''

    def __init__(self, out_channel_N=192, out_channel_M=192, one_lmbda=False):
        super(Analysis_prior_net, self).__init__()
        if one_lmbda:
            self.conv1 = nn.Conv2d(out_channel_M, out_channel_N, 3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
            self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        else:
            self.conv1 = ConditionalConv2d(out_channel_M, out_channel_N, 3, stride=1, padding=1)
            self.conv2 = ConditionalConv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
            self.conv3 = ConditionalConv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x, lmbda=None):
        if lmbda == None:
            x = torch.abs(x)
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            return self.conv3(x)
        else:
            x = torch.abs(x)
            x = self.relu1(self.conv1(x, lmbda))
            x = self.relu2(self.conv2(x, lmbda))
            return self.conv3(x, lmbda)


class Synthesis_prior_net(nn.Module):
    '''
    Decode synthesis prior
    '''

    def __init__(self, out_channel_N=192, out_channel_M=192, one_lmbda=False):
        super(Synthesis_prior_net, self).__init__()
        if one_lmbda:
            self.deconv1 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
            self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
            self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_M, 3, stride=1, padding=1)
        else:
            self.deconv1 = ConditionalDeconv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
            self.deconv2 = ConditionalDeconv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
            self.deconv3 = ConditionalDeconv2d(out_channel_N, out_channel_M, 3, stride=1, padding=1)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x, lmbda=None):
        if lmbda == None:
            x = self.relu1(self.deconv1(x))
            x = self.relu2(self.deconv2(x))
            return torch.exp(self.deconv3(x))
        else:
            x = self.relu1(self.deconv1(x, lmbda))
            x = self.relu2(self.deconv2(x, lmbda))
            return torch.exp(self.deconv3(x, lmbda))


class Synthesis_net(nn.Module):
    '''
    Decode synthesis
    '''

    def __init__(self, out_channel_N=192, out_channel_M=576, one_lmbda=False):
        super(Synthesis_net, self).__init__()
        if one_lmbda:
            self.deconv1 = nn.ConvTranspose2d(out_channel_M, out_channel_N, 5, stride=2, padding=2, output_padding=1)
            self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
            self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
            self.deconv4 = nn.ConvTranspose2d(out_channel_N, 3, 5, stride=2, padding=2, output_padding=1)
        else:
            self.deconv1 = ConditionalDeconv2d(out_channel_M, out_channel_N, 5, stride=2, padding=2, output_padding=1)
            self.deconv2 = ConditionalDeconv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
            self.deconv3 = ConditionalDeconv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
            self.deconv4 = ConditionalDeconv2d(out_channel_N, 3, 5, stride=2, padding=2, output_padding=1)

        self.igdn1 = GDN(out_channel_N, inverse=True)
        self.igdn2 = GDN(out_channel_N, inverse=True)
        self.igdn3 = GDN(out_channel_N, inverse=True)

    def forward(self, x, lmbda=None):
        if lmbda == None:
            x = self.igdn1(self.deconv1(x))
            x = self.igdn2(self.deconv2(x))
            x = self.igdn3(self.deconv3(x))
            x = self.deconv4(x)
        else:
            x = self.igdn1(self.deconv1(x, lmbda))
            x = self.igdn2(self.deconv2(x, lmbda))
            x = self.igdn3(self.deconv3(x, lmbda))
            x = self.deconv4(x, lmbda)
        return x


class Analysis_net(nn.Module):
    '''
    Analysis net
    '''

    def __init__(self, out_channel_N=192, out_channel_M=384, one_lmbda=False):
        super(Analysis_net, self).__init__()
        if one_lmbda:
            self.conv1 = nn.Conv2d(3, out_channel_N, 5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
            self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
            self.conv4 = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)
        else:
            self.conv1 = ConditionalConv2d(3, out_channel_N, 5, stride=2, padding=2)
            self.conv2 = ConditionalConv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
            self.conv3 = ConditionalConv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
            self.conv4 = ConditionalConv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)

        self.gdn1 = GDN(out_channel_N)
        self.gdn2 = GDN(out_channel_N)
        self.gdn3 = GDN(out_channel_N)
        self.ReLU = nn.LeakyReLU()

        # 중간에 뻗어나가는 가지

    def forward(self, x, lmbda=None, stem=False):
        if lmbda == None:
            x = self.gdn1(self.conv1(x))
            x = self.gdn2(self.conv2(x))
            x = self.gdn3(self.conv3(x))
            if stem:
                return self.ReLU(self.stem(x))
            else:
                return self.conv4(x)
        else:
            x = self.gdn1(self.conv1(x, lmbda))
            x = self.gdn2(self.conv2(x, lmbda))
            x = self.gdn3(self.conv3(x, lmbda))
            if stem:
                return self.ReLU(self.stem(x, lmbda))
            else:
                return self.conv4(x, lmbda)


class Bitparm(nn.Module):
    '''
    save params
    '''

    def __init__(self, channel, final=False):
        super(Bitparm, self).__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        else:
            self.a = None

    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)


class BitEstimator(nn.Module):
    '''
    Estimate bit
    '''

    def __init__(self, channel):
        super(BitEstimator, self).__init__()
        self.f1 = Bitparm(channel)
        self.f2 = Bitparm(channel)
        self.f3 = Bitparm(channel)
        self.f4 = Bitparm(channel, True)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)


class EntropyParamsZ(nn.Module):
    '''
    Estimate bit
    '''

    def __init__(self):
        super(EntropyParamsZ, self).__init__()
        self.conv1 = ConditionalConv2d(384, 640, 1, stride=1, padding=0)
        self.ReLU1 = nn.LeakyReLU()

        self.conv2 = ConditionalConv2d(640, 512, 1, stride=1, padding=0)
        self.ReLU2 = nn.LeakyReLU()

        self.stem1 = ConditionalConv2d(256, 192, 1, stride=1, padding=0)

        self.stem2 = ConditionalConv2d(256, 192, 1, stride=1, padding=0)

        self.ReLU = nn.ReLU()

    def forward(self, context_z, lmbda):
        context_z = self.ReLU1(self.conv1(context_z, lmbda))
        context_z = self.ReLU2(self.conv2(context_z, lmbda))
        scale, mean = context_z.chunk(2, 1)
        scale = self.ReLU(self.stem1(scale, lmbda))
        mean = self.stem2(mean, lmbda)
        return scale, mean


class EntropyParameters_Samsung(nn.Module):
    '''
    Analysis prior net
    '''

    def __init__(self):
        super(EntropyParameters_Samsung, self).__init__()
        self.conv1 = ConditionalDeconv2d(192, 384, 5, stride=2, padding=2, output_padding=1)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.conv2 = ConditionalDeconv2d(384, 576, 5, stride=2, padding=2, output_padding=1)
        self.relu2 = nn.LeakyReLU(inplace=True)

        self.conv3 = ConditionalDeconv2d(576, 768, 3, stride=1, padding=1)

        self.conv4 = ConditionalConv2d(1536, 1280, 1, stride=1, padding=0)
        self.relu4 = nn.LeakyReLU(inplace=True)

        self.conv5 = ConditionalConv2d(1280, 1024, 1, stride=1, padding=0)
        self.relu5 = nn.LeakyReLU(inplace=True)

        self.stem1 = ConditionalConv2d(512, 384, 1, stride=1, padding=0)
        self.stem2 = ConditionalConv2d(512, 384, 1, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, z_hat, y_context, lmbda):
        params = self.relu1(self.conv1(z_hat, lmbda))
        params = self.relu2(self.conv2(params, lmbda))
        params = self.conv3(params, lmbda)
        params = torch.cat([params, y_context], dim=1)
        params = self.relu4(self.conv4(params, lmbda))
        params = self.relu5(self.conv5(params, lmbda))
        scale, mean = params.chunk(2, 1)
        scale = self.relu(self.stem1(scale, lmbda))
        mean = self.stem2(mean, lmbda)

        return scale, mean


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blured
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blured tensors
    """

    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out


def _ssim(X, Y, win, data_range=255, size_average=True, full=False):
    r""" Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
    Returns:
        torch.Tensor: ssim results
    """

    K1 = 0.01
    K2 = 0.03
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=255, size_average=True, full=False):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
    Returns:
        torch.Tensor: ssim results
    """

    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    ssim_val, cs = _ssim(X, Y,
                         win=win,
                         data_range=data_range,
                         size_average=False,
                         full=True)
    if size_average:
        ssim_val = ssim_val.mean()
        cs = cs.mean()

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ms_ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=255, size_average=True, full=False, weights=None):
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
        weights (list, optional): weights for different levels
    Returns:
        torch.Tensor: ms-ssim results
    """
    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if weights is None:
        weights = torch.FloatTensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(X.device, dtype=X.dtype)

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    levels = weights.shape[0]
    mcs = []
    for _ in range(levels):
        ssim_val, cs = _ssim(X, Y,
                             win=win,
                             data_range=data_range,
                             size_average=False,
                             full=True)
        mcs.append(cs)

        padding = (X.shape[2] % 2, X.shape[3] % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    mcs = torch.stack(mcs, dim=0)  # mcs, (level, batch)
    # weights, (level)
    msssim_val = torch.prod((mcs[:-1] ** weights[:-1].unsqueeze(1))
                            * (ssim_val ** weights[-1]), dim=0)  # (batch, )

    if size_average:
        msssim_val = msssim_val.mean()
    return msssim_val


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=None, size_average=True, channel=3):
        r""" class for ssim
        Args:
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            channel (int, optional): input channels (default: 3)
        """

        super(SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range

    def forward(self, X, Y):
        return ssim(X, Y, win=self.win, data_range=self.data_range, size_average=self.size_average)


class MS_SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=None, size_average=True, channel=3, weights=None):
        r""" class for ms-ssim
        Args:
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
        """

        super(MS_SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights

    def forward(self, X, Y):
        return ms_ssim(X, Y, win=self.win, size_average=self.size_average, data_range=self.data_range,
                       weights=self.weights)


class VideoFolder(Dataset):

    def __init__(self, root, mode='train', transform=None):
        from tqdm import tqdm
        self.mode = mode
        root_dir = Path(root)
        self.transform = transform

        if self.mode == 'train':
            from random import sample
            self.samples = []
            for sub_f in tqdm(root_dir.iterdir()):
                if sub_f.is_dir():
                    for sub_sub_f in Path(sub_f).iterdir():
                        for i in range(5):
                            # print(sample(list(sub_sub_f.iterdir()), k=2))
                            self.samples.append(sample(list(sub_sub_f.iterdir()), k=2))

            if not root_dir.is_dir():
                raise RuntimeError(f'Invalid directory "{root}"')

            # self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        else:
            self.samples = {}
            self.video_names = []
            for sub_f in tqdm(root_dir.iterdir()):
                if sub_f.is_dir():
                    self.video_names.append(Path(sub_f).name)
                    self.samples[Path(sub_f).name] = [img for img in Path(sub_f).iterdir() if img.is_file()]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        if self.mode == 'train':
            try:
                imgs = self.samples[index]
                img1 = Image.open(imgs[0]).convert("RGB")
                img2 = Image.open(imgs[1]).convert("RGB")
                if self.transform:
                    return [self.transform(img1), self.transform(img2)]
                else:
                    return [img1, img2]
            except:
                pass
        # test
        else:
            video_name = self.video_names[index]
            imgs = self.samples[video_name]
            if self.transform:
                return (video_name, [self.transform(Image.open(img).convert("RGB")) for img in imgs])
            else:
                return imgs

    def __len__(self):
        return len(self.samples)

class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        # print(self.samples)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        try:
            img = Image.open(self.samples[index]).convert("RGB")
            if self.transform:
                # print(type(self.transform(img)))
                return self.transform(img)
            return img
        except:
            pass

    def __len__(self):
        return len(self.samples)