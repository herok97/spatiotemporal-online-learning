import math
import random

import time
from random import choice, choices

import cv2
import cv2 as cv
import numpy as np
import torch
from tqdm import tqdm, trange
from ImageCompressor_Minnen_CConv import ImageCompressor_Minnen_CConv
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.datasets import ImageFolder
from basics import *
# 텐서보드 추가
from tensorboardX import SummaryWriter


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

class Solver():
    def __init__(self, config, isTrain=True):
        self.imageCompressor = ImageCompressor_Minnen_CConv(192, 320).cuda()
        self.config = config
        self.global_step = 0
        self.isTrain = isTrain

    def load_dataset(self):

        # DataLoader
        if self.isTrain:
            print('load train datasets')
            train_transforms = transforms.Compose(
                [
                    transforms.ToTensor()
                ]
            )

            train_dataset = ImageFolder(self.config.dataset, split="train", transform=train_transforms)

            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                shuffle=True,
                pin_memory=True
            )

        test_transforms = transforms.Compose(
            [transforms.ToTensor()]
        )

        test_dataset = ImageFolder(self.config.dataset, split="test", transform=test_transforms)

        print('load test datasets')

        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config.test_batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=True,
        )
    def fix_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True   # adaptivepooling 사용시 RuntimeError
        torch.backends.cudnn.benchmark = False         # False로 해야 seed 고정이라고 하는데 False로 하면 느린 것 같음
        np.random.seed(seed)
        random.seed(seed)

    def build(self):
        self.fix_seed(self.config.seed)
        parameters = self.imageCompressor.parameters()
        self.optimizer = optim.Adam(parameters, lr=self.config.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1000000, 1100000, 1150000, 1200000], gamma=0.5)
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[800000, 1050000, 1150000, 1200000], gamma=0.5)

        if self.config.pre_train:
            print(f'load pre-trained model {self.config.save_model_dir}')
            if self.isTrain:
                self.load_checkpoint(self.imageCompressor,
                                     [self.optimizer],
                                     [self.scheduler],
                                     self.config.save_model_dir)
            else:
                self.load_model(self.imageCompressor, self.config.save_model_dir)
        # self.scheduler.milestones=[1600000, 2100000, 2300000, 2400000]
        self.optimizer.param_groups[0]['lr'] = 0.00003 * 0.5
        # 데이터셋 로드
        self.load_dataset()

    def one_hot(self, lmbda):
        lmbda_one_hot = torch.tensor([0.] * len(self.config.lmbda)).cuda()
        lmbda_index = self.config.lmbda.index(lmbda)
        lmbda_one_hot[lmbda_index] = 1.0
        return lmbda_one_hot

    def train(self):
        self.writer = SummaryWriter(self.config.log_dir)
        self.imageCompressor.train()

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

                # for lm in range(5):
                #     # forward network
                #     original_img = img.cuda()
                #     # lambda 값은 순차적으로 접근 (균일하게 하기 위해서)
                #     lmbda = self.config.lmbda[lm % 5]
                #     lmbda_one_hot = self.one_hot(lmbda)  # for multiple bitrate
                #
                #     x_hat, y_hat, mse_loss, bpp_feature, bpp_z, bpp = self.imageCompressor(original_img,
                #                                                                            lmbda_one_hot)
                #     mse_loss, bpp_feature, bpp_z, bpp = \
                #         torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
                #     rd_loss = mse_loss * lmbda + bpp
                #     if lm == 0:
                #         total_loss = rd_loss
                #     else:
                #         total_loss += rd_loss
                #
                #     psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10))
                #
                #     cum_bpp[str(lmbda)].append(np.float(bpp.detach()))
                #     cum_loss[str(lmbda)].append(np.float(rd_loss.detach()))
                #     cum_psnr[str(lmbda)].append(np.float(psnr.detach()))
                # rd_loss = torch.sum(total_loss)

                # forward network
                original_img = img.cuda()
                # lambda 값은 랜덤 값
                lmbda = self.config.lmbda[random.randint(0, 8)]
                lmbda_one_hot = self.one_hot(lmbda)  # for multiple bitrate

                x_hat, y_hat, mse_loss, bpp_feature, bpp_z, bpp = self.imageCompressor(original_img,
                                                                                       lmbda_one_hot)
                mse_loss, bpp_feature, bpp_z, bpp = \
                    torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)

                rd_loss = mse_loss * lmbda + bpp

                self.optimizer.zero_grad()
                rd_loss.backward()

                clip_gradient(self.optimizer, self.config.clip_max_norm)

                self.optimizer.step()
                self.scheduler.step()

                psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10))

                cum_bpp[str(lmbda)].append(np.float(bpp.detach()))
                cum_loss[str(lmbda)].append(np.float(rd_loss.detach()))
                cum_psnr[str(lmbda)].append(np.float(psnr.detach()))


                if self.global_step % 50 == 0:
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
                    self.save_checkpoint(epoch, self.imageCompressor, [self.optimizer], [self.scheduler],
                                         self.global_step,
                                         self.config.save_dir + str(time.time()).split('.')[0] + '_' + str(
                                             self.global_step) + '.pkl')
                    print('----------------------TEST--------------------\n')
                    for lmbda_ in self.config.lmbda:
                        self.test(lmbda_)


    def test(self, lmbda):
        with torch.no_grad():
            lmbda_one_hot = self.one_hot(lmbda)  # for multiple bitrate

            self.imageCompressor.eval()
            sumBpp = 0
            sumPsnr = 0
            sumMsssim = 0
            sumMsssimDB = 0
            cnt = 0
            for batch_idx, input in enumerate(self.test_dataloader):
                input = input.cuda()
                recon_image, compressed_feature_renorm, mse_loss, bpp_feature, bpp_z, bpp = self.imageCompressor(input,
                                                                                                                 lmbda_one_hot)
                mse_loss, bpp_feature, bpp_z, bpp = \
                    torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
                psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10))
                sumBpp += bpp
                sumPsnr += psnr
                msssim = ms_ssim(recon_image.cpu().detach(), input.cpu().detach(), data_range=1.0, size_average=True)
                msssimDB = -10 * (torch.log(1 - msssim) / np.log(10))
                sumMsssimDB += msssimDB
                sumMsssim += msssim
                cnt += 1
                print(
                    "Dataset result---Lambda: {}, Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(
                        lmbda, bpp, psnr, msssim, msssimDB))
            sumBpp /= cnt
            sumPsnr /= cnt
            sumMsssim /= cnt
            sumMsssimDB /= cnt
            print(
                "Dataset Average result---Lambda: {}, Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(
                    lmbda, sumBpp, sumPsnr, sumMsssim, sumMsssimDB))
        self.imageCompressor.train()


    def save_checkpoint(self, epoch, model, optimizers, schedulers, global_step, path):
        os = []
        for optimizer in optimizers:
            os.append(optimizer.state_dict())
        ss = []
        for scheduler in schedulers:
            ss.append(scheduler.state_dict())
        state = {
            'Epoch': epoch,
            'State_dict': model.state_dict(),
            'optimizers': os,
            'scheduler': ss,
            'Global_step': global_step
        }
        torch.save(state, path)

    def load_checkpoint(self, model, optimizers, schedulers, path):
        checkpoint = torch.load(path)
        pretrained_dict = checkpoint['State_dict']
        new_model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
        print('Load layers\n', pretrained_dict.keys())
        new_model_dict.update(pretrained_dict)
        print('All layers\n', [name for name, _ in model.named_parameters()])
        model.load_state_dict(new_model_dict)

        for i, optimizer in enumerate(optimizers):
            optimizer.load_state_dict(checkpoint['optimizers'][i])
        for i, scheduler in enumerate(schedulers):
            scheduler.load_state_dict(checkpoint['scheduler'][i])
        self.global_step = checkpoint['Global_step']

    def load_model(self, model, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['State_dict'])
