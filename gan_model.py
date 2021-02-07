import os

import cv2
import numpy as np
import torch
from torch.nn import init

from histogram import CPLoss, GPLoss, HistogramLoss
from model_modules import Generator


class GANModel:
    def __init__(self, args):
        self.start_epoch = 0
        self.args = args

        self.G = Generator()
        # self.histogram_loss = CPLoss(rgb=False, yuvgrad=True)
        self.histogram_loss = HistogramLoss(
            loss_fn=args.hist_loss, rgb=False, yuvgrad=args.yuvgrad, num_bins=256
        )

        self.init_type = args.init_type
        if args.init_type is not None:
            self.G.apply(self.init_weights)

        self.optimizer_G = torch.optim.Adam(
            self.G.parameters(), lr=args.lr, betas=(args.beta1, 0.999)
        )

        self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=self.lr_lambda
        )

        self.gan_loss = GPLoss()
        self.lambda_g = args.lambda_g
        self.lambda_mi = args.lambda_mi
        self.lambda_emd = args.lambda_emd

    def lr_lambda(self, epoch):
        return 1.0 - max(0, epoch + self.start_epoch - self.args.lr_decay_start) / (
            self.args.lr_decay_n + 1
        )

    def init_weights(self, m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if self.init_type == "normal":
                init.normal_(m.weight.data, 0.0, 0.02)
            elif self.init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=0.02)
            elif self.init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            else:
                raise NotImplementedError(
                    "initialization method [%s] not implemented" % self.init_type
                )
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    def update_scheduler(self):
        self.scheduler_G.step()
        print("learning rate = %.7f" % self.optimizer_G.param_groups[0]["lr"])

    def set_start_epoch(self, epoch):
        self.start_epoch = epoch

    def to(self, device):
        self.G.to(device)
        self.histogram_loss.histlayer.to(device)

    def train(self, input, save, out_dir_img, epoch, i):
        edge, img, img_idx = input
        # convert list of one_d histogram into a tensor of multi-channel histogram
        histogram = torch.stack(
            self.histogram_loss.extract_hist(
                self.histogram_loss.to_YUV(img), one_d=True
            ),
            1,
        )

        ############################
        # G loss
        ############################
        self.optimizer_G.zero_grad()
        gen = self.G(edge, histogram)

        # GAN loss of G
        loss_G_gan = self.gan_loss(gen, img)

        # histogram loss
        emd_loss, mi_loss = self.histogram_loss(img, gen)
        loss_hist = emd_loss + mi_loss

        # Combine
        loss_G = (
            (self.lambda_g * loss_G_gan)
            + (self.lambda_emd * emd_loss)
            + (self.lambda_mi * mi_loss)
        )

        loss_G.backward()
        self.optimizer_G.step()

        # save image
        if save:
            self.save_image(
                (edge[0].unsqueeze(0), img[0].unsqueeze(0), gen[0].unsqueeze(0)),
                out_dir_img,
                "train_ep_%d_img_%d" % (epoch, img_idx[0]),
            )

        return {
            "G": loss_G,
            "G_gan": loss_G_gan,
            "G_H": loss_hist,
            "MI": mi_loss,
            "EMD": emd_loss,
        }

    def eval(self, input, save, out_dir_img, epoch):
        with torch.no_grad():
            edge, img, img_idx = input
            # convert list of one_d histogram into a tensor of multi-channel histogram
            histogram = torch.stack(
                self.histogram_loss.extract_hist(
                    self.histogram_loss.to_YUV(img), one_d=True
                ),
                1,
            )
            gen = self.G(edge, histogram)

            ############################
            # G loss
            ############################
            # GAN loss of G
            loss_G_gan = self.gan_loss(gen, img)

            # histogram loss
            emd_loss, mi_loss = self.histogram_loss(img, gen)
            loss_hist = emd_loss + mi_loss

            # Combine
            loss_G = (
                (self.lambda_g * loss_G_gan)
                + (self.lambda_emd * emd_loss)
                + (self.lambda_mi * mi_loss)
            )

        # save image
        if save:
            self.save_image(
                (edge[0].unsqueeze(0), img[0].unsqueeze(0), gen[0].unsqueeze(0)),
                out_dir_img,
                "val_ep_%d_img_%d" % (epoch, img_idx[0]),
            )

        return {
            "G": loss_G,
            "G_gan": loss_G_gan,
            "G_H": loss_hist,
            "MI": mi_loss,
            "EMD": emd_loss,
        }

    def test(self, images, i, out_dir_img):
        with torch.no_grad():
            A, B, img_idx, C = images
            # convert list of one_d histogram into a tensor of multi-channel histogram
            C = C if C is not None else B
            histogram = torch.stack(
                self.histogram_loss.extract_hist(
                    self.histogram_loss.to_YUV(C), one_d=True
                ),
                1,
            )
            gen = self.G(A, histogram)
            self.save_image(
                (A, B, gen), out_dir_img, "test_%d" % img_idx, test=True,
            )
        return 0, 0

    def load_state(self, state, lr=None):
        print("Using pretrained model...")
        self.G.load_state_dict(state["G"])
        self.optimizer_G.load_state_dict(state["optimG"])

        # set model lr to new lr
        if lr is not None:
            for param_group in self.optimizer_G.param_groups:
                before = param_group["lr"]
                param_group["lr"] = lr
            print("optim lr: before={} / after={}".format(before, lr))

    def save_state(self):
        return {
            "G": self.G.state_dict(),
            "optimG": self.optimizer_G.state_dict(),
        }

    def save_image(self, input, filepath, fname, test=False):
        """ input is a tuple of the images we want to compare """
        A, B, gen = input

        if test:
            img = self.tensor2image(gen)
            img = img.squeeze().transpose(1, 2, 0)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            path = os.path.join(filepath, "%s.png" % fname)
            cv2.imwrite(path, img)

        else:
            merged = self.tensor2image(self.merge_images(A, B, gen))
            merged = cv2.cvtColor(merged, cv2.COLOR_RGB2BGR)
            path = os.path.join(filepath, "%s.png" % fname)
            cv2.imwrite(path, merged)

        print("saved %s" % path)

    def tensor2image(self, input):
        image_data = input.data
        image = 127.5 * (image_data.cpu().float().numpy() + 1.0)
        return image.astype(np.uint8)

    def merge_images(self, sources, targets, generated):
        row, _, h, w = sources.size()
        merged = torch.zeros([3, row * h, w * 3])
        for idx, (s, t, g) in enumerate(zip(sources, targets, generated)):
            i = idx
            merged[:, i * h : (i + 1) * h, 0:w] = s
            merged[:, i * h : (i + 1) * h, w : 2 * w] = g
            merged[:, i * h : (i + 1) * h, 2 * w : 3 * w] = t
        return merged.permute(1, 2, 0)
