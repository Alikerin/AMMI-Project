import os

import numpy as np
import torch
from PIL import Image
from torch.nn import init

from histogram import HistogramLoss
from model_modules import Generator


class GANModel:
    def __init__(self, args):
        self.start_epoch = 0
        self.args = args

        self.G = Generator()
        # self.cp_loss = CPLoss(rgb=True, yuvgrad=True)
        self.cp_loss = HistogramLoss(
            loss_fn=args.hist_loss, rgb=False, yuv=True, num_bins=256
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

        self.gp_loss = torch.nn.MSELoss()  # GPLoss()
        self.lambda_g = args.lambda_g
        self.lambda_h = args.lambda_h

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
        self.cp_loss.histlayer.to(device)

    def train(self, input, save, out_dir_img, epoch, i):
        edge, content_ref, color_ref, img_idx = input
        # convert list of one_d histogram into a tensor of multi-channel histogram
        histogram = torch.stack(
            self.cp_loss.extract_hist(
                self.cp_loss.to_YUV(color_ref), one_d=True, normalize=True
            ),
            1,
        )

        ############################
        # G loss
        ############################
        self.optimizer_G.zero_grad()
        gen = self.G(edge, histogram)

        # GP loss
        loss_GP = self.lambda_g * self.gp_loss(gen, edge)  # since edge is grayscale

        # CP loss
        loss_CP = self.lambda_h * self.cp_loss(gen, color_ref)

        # SPL
        loss_SPL = loss_GP + loss_CP

        loss_SPL.backward()
        self.optimizer_G.step()

        # save image
        if save:
            self.save_image(
                (
                    edge[0].unsqueeze(0),
                    content_ref[0].unsqueeze(0),
                    color_ref[0].unsqueeze(0),
                    gen[0].unsqueeze(0),
                ),
                out_dir_img,
                "train_ep_%d_img_%d" % (epoch, img_idx[0]),
            )

        return {
            "loss_GP": loss_GP,
            "loss_CP": loss_CP,
            "loss_SPL": loss_SPL,
        }

    def eval(self, input, save, out_dir_img, epoch):
        with torch.no_grad():
            edge, content_ref, color_ref, img_idx = input
            # convert list of one_d histogram into a tensor of multi-channel histogram
            histogram = torch.stack(
                self.cp_loss.extract_hist(
                    self.cp_loss.to_YUV(color_ref), one_d=True, normalize=True
                ),
                1,
            )
            gen = self.G(edge, histogram)

            # GP loss
            loss_GP = self.gp_loss(gen, edge)  # since edge is grayscale

            # CP loss
            loss_CP = self.lambda_h * self.cp_loss(gen, color_ref)

            # SPL
            loss_SPL = loss_GP + loss_CP

        # save image
        if save:
            self.save_image(
                (
                    edge[0].unsqueeze(0),
                    content_ref[0].unsqueeze(0),
                    color_ref[0].unsqueeze(0),
                    gen[0].unsqueeze(0),
                ),
                out_dir_img,
                "val_ep_%d_img_%d" % (epoch, img_idx[0]),
            )

        return {
            "loss_GP": loss_GP,
            "loss_CP": loss_CP,
            "loss_SPL": loss_SPL,
        }

    def test(self, images, i, out_dir_img):
        with torch.no_grad():
            edge, content_ref, color_ref, img_idx = input
            # convert list of one_d histogram into a tensor of multi-channel histogram
            histogram = torch.stack(
                self.cp_loss.extract_hist(
                    self.cp_loss.to_YUV(color_ref), one_d=True, normalize=True
                ),
                1,
            )
            gen = self.G(edge, histogram)
            self.save_image(
                (edge, content_ref, color_ref, gen),
                out_dir_img,
                "test_%d" % img_idx,
                test=True,
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
        edge, content_ref, color_ref, gen = input

        if test:
            img = self.tensor2image(gen)
            img = img.squeeze().transpose(1, 2, 0)
            img = Image.fromarray(img)
            path = os.path.join(filepath, "%s.png" % fname)
            img.save(path)

        else:
            merged = self.merge_images(
                self.tensor2image(edge),
                self.tensor2image(content_ref),
                self.tensor2image(color_ref),
                self.tensor2image(gen),
            )
            path = os.path.join(filepath, "%s.png" % fname)
            merged.save(path)

        print("saved %s" % path)

    def tensor2image(self, input):
        image_data = input.data
        image = 127.5 * (image_data.cpu().float().numpy() + 1.0)
        return image.astype(np.uint8)

    def merge_images(self, edge, content_ref, color_ref, gen):
        print(edge.shape)
        _, _, h, w = edge.shape
        merged = Image.new("RGB", (w * 4, h))
        merged.paste(Image.fromarray(edge[0].transpose(1, 2, 0)), (0, 0))
        merged.paste(Image.fromarray(content_ref[0].transpose(1, 2, 0)), (w, 0))
        merged.paste(Image.fromarray(color_ref[0].transpose(1, 2, 0)), (w * 2, 0))
        merged.paste(Image.fromarray(gen[0].transpose(1, 2, 0)), (w * 3, 0))
        return merged
