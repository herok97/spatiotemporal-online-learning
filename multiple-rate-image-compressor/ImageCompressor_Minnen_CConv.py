from basics import *
import math

class mySequential(nn.Sequential):
    def forward(self, *input):
        input, lmbda = input
        for module in self._modules.values():
            if 'Conditional' in str(type(module)):
                input = module(input, lmbda)
            else:
                input = module(input)
        return input

class ImageCompressor_Minnen_CConv(nn.Module):
    def __init__(self, N=192, M=192):
        super(ImageCompressor_Minnen_CConv, self).__init__()
        self.N = N
        self.M = M
        self.Encoder = mySequential(
            ConditionalConv2d(3, self.N, 5, stride=2, padding=2),
            GDN(self.N),
            ConditionalConv2d(self.N, self.N, 5, stride=2, padding=2),
            GDN(self.N),
            ConditionalConv2d(self.N, self.N, 5, stride=2, padding=2),
            GDN(self.N),
            ConditionalConv2d(self.N, self.M, 5, stride=2, padding=2),
        )
        self.Decoder = mySequential(
            ConditionalDeconv2d(self.M, self.N, 5, stride=2, padding=2, output_padding=1),
            GDN(self.N, inverse=True),
            ConditionalDeconv2d(self.N, self.N, 5, stride=2, padding=2, output_padding=1),
            GDN(self.N, inverse=True),
            ConditionalDeconv2d(self.N, self.N, 5, stride=2, padding=2, output_padding=1),
            GDN(self.N, inverse=True),
            ConditionalDeconv2d(self.N, 3, 5, stride=2, padding=2, output_padding=1)
        )
        self.priorEncoder = mySequential(
            ConditionalConv2d(self.M, self.N, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            ConditionalConv2d(self.N, self.N, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            ConditionalConv2d(self.N, self.N, 5, stride=2, padding=2)
        )
        self.priorDecoder = mySequential(
            ConditionalDeconv2d(self.N, self.N, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            ConditionalDeconv2d(self.N, self.N * 3 // 2, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            ConditionalDeconv2d(self.N * 3 // 2, self.M * 2, 3, stride=1, padding=1)
        )

        self.bitEstimator_z = BitEstimator(self.N)

        self.Context = ConditionalMaskedConv2d(self.M, self.M * 2, kernel_size=5, padding=2, stride=1)

        self.entropy_parameters = mySequential(
            ConditionalConv2d(self.M * 12 // 3, self.M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            ConditionalConv2d(self.M * 10 // 3, self.M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            ConditionalConv2d(self.M * 8 // 3, self.M * 6 // 3, 1)
        )

    def forward(self, input_image, lmbda):
        quant_noise_feature = torch.zeros(input_image.size(0), self.M, input_image.size(2) // 16,
                                          input_image.size(3) // 16).cuda()
        quant_noise_z = torch.zeros(input_image.size(0), self.N, input_image.size(2) // 64,
                                    input_image.size(3) // 64).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -0.5, 0.5)

        y = self.Encoder(input_image, lmbda)
        z = self.priorEncoder(y, lmbda)
        batch_size = y.size()[0]

        if self.training:
            y_hat = y + quant_noise_feature
            z_hat = z + quant_noise_z
        else:
            y_hat = torch.round(y)
            z_hat = torch.round(z)

        # LVC_2021_Zhizheng_Learned Block-based Hybrid Image Compression
        y_hat_prime = y - (y - torch.round(y)).detach()

        sigma = self.priorDecoder(z_hat, lmbda)
        x_hat = self.Decoder(y_hat_prime, lmbda)

        ctx_params = self.Context(y_hat, lmbda)

        gaussian_params = self.entropy_parameters(torch.cat((sigma, ctx_params), dim=1), lmbda)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        # distortion
        mse_loss = torch.mean((x_hat - input_image).pow(2))

        def feature_probs_based_sigma(feature, scale, mu):
            gaussian = torch.distributions.laplace.Laplace(mu, torch.exp(scale).clamp(1e-10, 1e10))
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, probs

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob

        total_bits_feature, _ = feature_probs_based_sigma(y_hat, scales_hat, means_hat)
        total_bits_z, _ = iclr18_estimate_bits_z(z_hat)
        im_shape = input_image.size()
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z
        return x_hat, y_hat, mse_loss, bpp_feature, bpp_z, bpp


if __name__ == '__main__':
    # x = torch.ones(5, 3, 256, 256).cuda()
    # net = ImageCompressor_Minnen().cuda()
    # # lmbda = torch.tensor([0.] * 5).cuda()
    # # lmbda[0] = 1.
    # # lmbda = None
    # recon_y_hat, y_hat, mse_loss, bpp_feature, bpp_z, bpp = net(x)
    # print(recon_y_hat.shape, y_hat.shape, bpp_feature, bpp_z, bpp)
    pass
