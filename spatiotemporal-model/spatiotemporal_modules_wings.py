from basics import BitEstimator, ConditionalConv2d, ConditionalDeconv2d
import torch
from torch import nn
import math


class PhNet(nn.Module):
    def __init__(self, N=384, M=320):
        super(PhNet, self).__init__()
        self.N = N
        self.M = M
        self.Encoder = PhEncoder(N, M)
        self.Decoder = PhDecoer(N, M)
        self.bitEstimator_z = BitEstimator(M)

    def forward(self, input):
        quant_noise_z = torch.zeros(input.size(0), self.M, input.size(2) // 4, input.size(3) // 4).cuda()
        quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -0.5, 0.5)
        z = self.Encoder(input)
        batch_size = input.size()[0]

        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)

        recon_z = self.Decoder(compressed_z)

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob

        total_bits_z, _ = iclr18_estimate_bits_z(compressed_z)
        im_shape = input.size()
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3] * 16 * 16)

        return recon_z, compressed_z, bpp_z


class PhEncoder(nn.Module):
    def __init__(self, N=192, M=256):
        super(PhEncoder, self).__init__()
        self.conv1 = ConditionalConv2d(2 * N, M, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.conv2 = ConditionalConv2d(M, M, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.conv3 = ConditionalConv2d(M, M, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)

        self.relu1 = nn.LeakyReLU(inplace=True)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.conv3(x)


class PhDecoer(nn.Module):
    def __init__(self, N=384, M=320):
        super(PhDecoer, self).__init__()
        self.conv1 = ConditionalDeconv2d(M, M, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.conv2 = ConditionalDeconv2d(M, M, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.conv3 = ConditionalDeconv2d(M, 2 * N, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.conv3(x)


class TemporalPrior(nn.Module):
    def __init__(self, N=384):
        super(TemporalPrior, self).__init__()
        self.conv1 = ConditionalConv2d(N, (N // 3) * 4, 5, stride=1, padding=2)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.conv2 = ConditionalConv2d((N // 3) * 4, (N // 3) * 5, 5, stride=1, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.conv3 = ConditionalConv2d((N // 3) * 5, N * 2, 5, stride=1, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)

        self.relu1 = nn.LeakyReLU(inplace=True)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.conv3(x)


class EntropyParam(nn.Module):
    def __init__(self, N=384):
        super(EntropyParam, self).__init__()
        self.conv1 = ConditionalConv2d(4 * N, 5 * N, 1, stride=1, padding=0)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.conv2 = ConditionalConv2d(5 * N, 4 * N, 1, stride=1, padding=0)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.conv3 = ConditionalConv2d(4 * N, 2 * N, 1, stride=1, padding=0)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)

        self.relu1 = nn.LeakyReLU(inplace=True)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.conv3(x)
