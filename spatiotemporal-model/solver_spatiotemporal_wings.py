import random, time
import torch
from torch import cuda, optim
from tqdm import tqdm
from VideoCompressor_wings import VideoCompressor_wings
from ImageCompressor_Minnen_CConv import ImageCompressor_Minnen_CConv
from basics import VideoFolder, ms_ssim
from torch.utils.data import DataLoader
# 텐서보드 추가
from tensorboardX import SummaryWriter
from torchvision import transforms
import numpy as np


class Solver():
    def __init__(self, config, isTrain=True):
        self.videoCompressor = VideoCompressor_wings(384, 320).cuda()
        self.imageCompressor = ImageCompressor_Minnen_CConv(256, 384).cuda()
        self.config = config
        self.isTrain = isTrain

        # load pre-trained ImgaeCompressor model
        self.load_model(self.imageCompressor, self.config.imgCompDir)

        self.global_step = 0

    def load_dataset(self):

        # DataLoader
        if self.isTrain:
            print('load train datasets')
            train_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.CenterCrop(256)
                ]
            )

            train_dataset = VideoFolder(self.config.train_dataset, transform=train_transforms)

            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                shuffle=True,
                pin_memory=True,
            )

        test_transforms = transforms.Compose(
            [transforms.Resize((1920, 1088)),
             transforms.ToTensor()]
        )

        test_dataset = VideoFolder(self.config.test_dataset, mode='test', transform=test_transforms)

        print('load test datasets')

        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config.test_batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def build(self):
        print('model build')

        # Initialize optimizer
        self.imageCompressor.eval()
        parameters = self.videoCompressor.parameters()
        self.optimizer = optim.Adam(parameters, lr=self.config.lr)
        self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=self.config.decay_step,
                                                   gamma=self.config.lr_decay)

        # pre-train model load
        if self.config.pre_train:
            if self.isTrain:
                print(f'load pre-trained model {self.config.save_model_dir}')
                self.load_checkpoint(self.videoCompressor, [self.optimizer], self.scheduler, self.config.save_model_dir)
            else:
                print(f'load pre-trained model {self.config.save_model_dir}')
                self.load_model(self.videoCompressor, self.config.save_model_dir)

        # 데이터셋 로드
        self.load_dataset()

        # fix seed
        self.fix_seed(self.config.seed)

        for name, param in self.videoCompressor.named_parameters():
            if '.ss' in name or '.bb' in name:
                print(name, 'grad=False')
                param.requires_grad = False

    def fix_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True   # adaptivepooling 사용시 RuntimeError
        torch.backends.cudnn.benchmark = False  # False로 해야 seed 고정이라고 하는데 False로 하면 느린 것 같음
        np.random.seed(seed)
        random.seed(seed)

    def one_hot(self, lmbda):
        lmbda_one_hot = torch.tensor([0.] * len(self.config.lmbda)).cuda()
        lmbda_index = self.config.lmbda.index(lmbda)
        lmbda_one_hot[lmbda_index] = 1.0
        return lmbda_one_hot

    def train(self):
        self.writer = SummaryWriter(self.config.log_dir)
        self.videoCompressor.train()
        #
        print('학습할 파라미터')
        for name, param in self.videoCompressor.named_parameters():
            if param.requires_grad:
                print(name)

        cum_bpp = {str(i): [] for i in self.config.lmbda}
        cum_psnr = {str(i): [] for i in self.config.lmbda}
        cum_loss = {str(i): [] for i in self.config.lmbda}

        for epoch in range(1, self.config.epochs + 1):
            tqdm.write(f'Epoch[{epoch}/{self.config.epochs}]')

            for i, img in enumerate(tqdm(self.train_dataloader)):
                # globalstep update
                self.global_step += 1

                # total global step을 사용해서도 break
                if self.config.total_global_step < self.global_step:
                    exit(0)

                for lm in range(5):
                    # forward network
                    img1, img2 = img[0].cuda(), img[1].cuda()
                    lmbda = self.config.lmbda[lm % 5]
                    lmbda_one_hot = self.one_hot(lmbda)  # for multiple bitrate

                    y_hat1 = self.imageCompressor.y_hat(img1, lmbda_one_hot)
                    y_hat2 = self.imageCompressor.y_hat(img2, lmbda_one_hot)

                    y_r, bpp_feature, bpp_z, bpp = self.videoCompressor(y_hat1, y_hat2)

                    y_recon = y_hat1 + y_r
                    recon_image = self.imageCompressor.Decoder(y_recon, lmbda_one_hot)
                    mse_loss = torch.mean((recon_image - img2).pow(2))

                    mse_loss, bpp_feature, bpp_z, bpp = \
                        torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)

                    rd_loss = bpp
                    if lm == 0:
                        total_loss = rd_loss
                    else:
                        total_loss += rd_loss

                    psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10))

                    cum_bpp[str(lmbda)].append(np.float(bpp.detach()))
                    cum_loss[str(lmbda)].append(np.float(rd_loss.detach()))
                    cum_psnr[str(lmbda)].append(np.float(psnr.detach()))

                def clip_gradient(optimizer, grad_clip):
                    for group in optimizer.param_groups:
                        for param in group["params"]:
                            if param.grad is not None:
                                param.grad.data.clamp_(-grad_clip, grad_clip)

                rd_loss = torch.sum(total_loss)
                self.optimizer.zero_grad()
                rd_loss.backward()

                clip_gradient(self.optimizer, self.config.clip_max_norm)
                self.optimizer.step()
                self.scheduler.step()

                if self.global_step % 25 == 0:
                    for i in self.config.lmbda:
                        cum_bpp[str(i)] = np.mean(cum_bpp[str(i)])
                        cum_loss[str(i)] = np.mean(cum_loss[str(i)])
                        cum_psnr[str(i)] = np.mean(cum_psnr[str(i)])

                    # write tensroboard
                    self.writer.add_scalars('bpp', cum_bpp, self.global_step)
                    self.writer.add_scalars('total_loss', cum_loss, self.global_step)
                    self.writer.add_scalars('psnr', cum_psnr, self.global_step)

                    tqdm.write(
                        f'Step:[{self.global_step}/{self.config.total_global_step}]\t' +
                        # f'lr: {self.optimizer.param_groups[0]["lr"]}\n' +
                        f'lr: {self.optimizer.param_groups[0]["lr"]}\n' +
                        f'bpp: {[np.round(i, 3) for i in cum_bpp.values()]}\n' +
                        f'psnr: {[np.round(i, 3) for i in cum_psnr.values()]}\n' +
                        f'loss: {[np.round(i, 3) for i in cum_loss.values()]}\n')

                    cum_bpp = {str(i): [] for i in self.config.lmbda}
                    cum_psnr = {str(i): [] for i in self.config.lmbda}
                    cum_loss = {str(i): [] for i in self.config.lmbda}

                if self.global_step % 10000 == 0:
                    del y_hat2, y_hat1, y_r, recon_image, img1, img2
                    torch.cuda.empty_cache()
                    self.save_checkpoint(epoch, self.videoCompressor, [self.optimizer], self.scheduler,
                                         self.global_step,
                                         self.config.save_dir + str(time.time()).split('.')[0] + '_' + str(
                                             self.global_step) + '.pkl')
                    # print('----------------------TEST--------------------\n')
                    # for lmbda_ in self.config.lmbda:
                    #     self.test(lmbda_)

    def test(self, lmbda):
        torch.cuda.empty_cache()
        with torch.no_grad():
            lmbda_one_hot = self.one_hot(lmbda)  # for multiple bitrate
            self.videoCompressor.eval()
            print("Test on UVG dataset")
            for batch_idx, input in enumerate(self.test_dataloader):
                sumBpp = 0
                sumPsnr = 0
                sumMsssim = 0
                sumMsssimDB = 0
                cnt = 0
                print(f"Video name: {input[0]}")
                for i, original_img in enumerate(input[1]):
                    img = original_img.cuda()
                    if i % 12 == 0:
                        x_hat, y_hat, mse_loss, bpp_feature, bpp_z, bpp = \
                            self.imageCompressor(img, lmbda_one_hot)
                    else:
                        y_hat_next = self.imageCompressor.y_hat(img, lmbda_one_hot)
                        y_r, bpp_feature, bpp_z, bpp = self.videoCompressor(y_hat, y_hat_next)
                        y_hat = y_hat + y_r
                        x_hat = self.imageCompressor.Decoder(y_hat, lmbda_one_hot)
                        mse_loss = torch.mean((x_hat - img).pow(2))

                    mse_loss, bpp_feature, bpp_z, bpp = \
                        torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
                    psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
                    sumBpp += bpp
                    sumPsnr += psnr
                    msssim = ms_ssim(x_hat.cpu().detach(), original_img.cpu().detach(), data_range=1.0,
                                     size_average=True)
                    msssimDB = -10 * (torch.log(1 - msssim) / np.log(10))
                    sumMsssimDB += msssimDB
                    sumMsssim += msssim
                    cnt += 1
                    print(
                        "Frame{}, lambda:{}, Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(i,
                                                                                                                lmbda,
                                                                                                                bpp,
                                                                                                                psnr,
                                                                                                                msssim,
                                                                                                                msssimDB))
                sumBpp /= cnt
                sumPsnr /= cnt
                sumMsssim /= cnt
                sumMsssimDB /= cnt
                print(
                    "Dataset Average result---,lambda:{}, Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(
                        lmbda, sumBpp, sumPsnr, sumMsssim, sumMsssimDB))
        self.videoCompressor.train()

    def test_(self):
        for name, param in self.videoCompressor.named_parameters():
            if 'entropy' not in name:
                param.requires_grad = False

        # original model weight
        self.videoCompressor.train()
        origin_weight = {}
        for name, param in self.videoCompressor.named_parameters():
            if 'entropy' in name:
                origin_weight[name] = param.data.clone().detach()

        print("Test on UVG dataset")
        for batch_idx, input in enumerate(self.test_dataloader):
            sumBpp = 0
            sumPsnr = 0
            sumMsssim = 0
            sumMsssimDB = 0
            cnt = 0
            print(f"Video name: {input[0]}")
            for i, original_img in enumerate(input[1]):

                img = original_img.cuda()
                self.videoCompressor.zero_grad()
                if i % 12 == 0:
                    x_hat, y_hat, mse_loss, bpp_feature, bpp_z, bpp = \
                        self.imageCompressor(img)
                else:
                    y_hat_next = self.imageCompressor.y_hat(img)
                    y_r, bpp_feature, bpp_z, bpp = self.videoCompressor(y_hat, y_hat_next)
                    y_hat = y_hat + y_r
                    x_hat = self.imageCompressor.Decoder(y_hat)
                    mse_loss = torch.mean((x_hat - img).pow(2))

                mse_loss, bpp_feature, bpp_z, bpp = \
                    torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)

                psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
                sumBpp += bpp
                sumPsnr += psnr
                msssim = ms_ssim(x_hat.cpu().detach(), original_img.cpu().detach(), data_range=1.0,
                                 size_average=True)
                msssimDB = -10 * (torch.log(1 - msssim) / np.log(10))
                sumMsssimDB += msssimDB
                sumMsssim += msssim
                cnt += 1
                print(
                    "Frame{}, Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(i,
                                                                                                 bpp, psnr,
                                                                                                 msssim,
                                                                                                 msssimDB))

            sumBpp /= cnt
            sumPsnr /= cnt
            sumMsssim /= cnt
            sumMsssimDB /= cnt
            print(
                "Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(
                    sumBpp, sumPsnr, sumMsssim, sumMsssimDB))

    def save_checkpoint(self, epoch, model, optimizers, scheduler, global_step, path):
        os = []
        for optimizer in optimizers:
            os.append(optimizer.state_dict())
        state = {
            'Epoch': epoch,
            'State_dict': model.state_dict(),
            'optimizers': os,
            'scheduler': scheduler.state_dict(),
            'Global_step': global_step
        }
        torch.save(state, path)

    def load_checkpoint(self, model, optimizers, scheduler, path):
        checkpoint = torch.load(path)
        pretrained_dict = checkpoint['State_dict']
        new_model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
        print('layer load\n', pretrained_dict.keys())
        new_model_dict.update(pretrained_dict)
        model.load_state_dict(new_model_dict)
        for i, optimizer in enumerate(optimizers):
            optimizer.load_state_dict(checkpoint['optimizers'][i])
        scheduler.load_state_dict(checkpoint['scheduler'])
        self.global_step = checkpoint['Global_step']

    def load_model(self, model, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['State_dict'])
