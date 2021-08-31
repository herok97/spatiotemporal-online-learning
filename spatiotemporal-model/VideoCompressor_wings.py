from basics import *
from spatiotemporal_modules_wings import *

class mySequential(nn.Sequential):
    def forward(self, *input):
        input, lmbda = input
        for module in self._modules.values():
            if 'Conditional' in str(type(module)):
                input = module(input, lmbda)
            else:
                input = module(input)
        return input

class VideoCompressor_wings(nn.Module):
    def __init__(self, N=384, M=320):
        super(VideoCompressor_wings, self).__init__()
        self.phM = PhNet(N, M)
        self.tpM = TemporalPrior(N)
        self.entropyParam = EntropyParam(N)

    def forward(self, y_p, y_c):
        # Residual
        y_r = y_c - y_p

        batch_size = y_r.size()[0]

        # PHE/D
        ph_input = torch.cat([y_p, y_c], dim=1)
        recon_z, _, bpp_z = self.phM(ph_input)  # can receive compressed_z

        # TPM
        tpM_output = self.tpM(y_p)

        # EPM
        epM_input = torch.cat([recon_z, tpM_output], dim=1)
        epM_output = self.entropyParam(epM_input)

        scales_hat, means_hat = epM_output.chunk(2, 1)

        def feature_probs_based_sigma(feature, scales_hat, means_hat):
            gaussian = torch.distributions.laplace.Laplace(means_hat, torch.exp(scales_hat).clamp(1e-10, 1e10))
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, probs

        total_bits_feature, _ = feature_probs_based_sigma(y_r, scales_hat, means_hat)
        im_shape = y_r.size()
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3] * 16 * 16)

        bpp = bpp_feature + bpp_z

        return y_r, bpp_feature, bpp_z, bpp


if __name__ == '__main__':
    x1 = torch.ones(5, 320, 16, 16).cuda()
    x2 = torch.ones(5, 320, 16, 16).cuda()
    net = VideoCompressor().cuda()
    # net.update()
    y_r, bpp_feature, bpp_z, bpp = net(x1, x2)
    print(y_r.shape, bpp_feature, bpp_z, bpp)
    # print(out.keys())s
