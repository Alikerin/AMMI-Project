import itertools
import os

import numpy as np
import torch
from PIL import Image
from torch.nn import init

from histogram import HistogramLoss
from model_modules import Discriminator, Generator


class GANModel:
    def __init__(self, args):
        self.start_epoch = 0
        self.args = args

        self.G = Generator()
        self.D = Discriminator()
        self.histogram_loss = HistogramLoss(
            loss_fn=args.hist_loss, num_bins=256, yuvgrad=args.yuvgrad
        )

        self.init_type = args.init_type
        if args.init_type is not None:
            self.G.apply(self.init_weights)
            self.D.apply(self.init_weights)

        self.optimizer_G = torch.optim.Adam(
            self.G.parameters(), lr=args.lr, betas=(args.beta1, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            self.D.parameters(), lr=args.lr, betas=(args.beta1, 0.999)
        )

        self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=self.lr_lambda
        )
        self.scheduler_D = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D, lr_lambda=self.lr_lambda
        )

        if args.gan_loss == "BCE":
            self.gan_loss_fn = torch.nn.BCELoss()
        elif args.gan_loss == "MSE":
            self.gan_loss_fn = torch.nn.MSELoss()
        else:
            raise NotImplementedError("GAN loss function error")

        self.lambd = args.lambd
        self.lambd_d = args.lambd_d
        self.lambda_mi = args.lambda_mi
        self.lambda_emd = args.lambda_emd

        self.d_update_frequency = args.d_update_frequency

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
        self.scheduler_D.step()
        print("learning rate = %.7f" % self.optimizer_G.param_groups[0]["lr"])

    def d_update(self, d_loss, epoch):
        # d_update_frequency = n epochs per update
        # d_update_epoch = list(range(1,300,int(1/self.d_update_frequency)))
        if epoch % self.d_update_frequency == 0:
            d_loss.backward()
            self.optimizer_D.step()

    def set_start_epoch(self, epoch):
        self.start_epoch = epoch

    def to(self, device):
        self.G.to(device)
        self.D.to(device)
        self.histogram_loss.histlayer.to(device)

        for state in itertools.chain(
            self.optimizer_G.state.values(), self.optimizer_D.state.values()
        ):
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def train(self, input, save, out_dir_img, epoch, i):
        # self.G.train()
        # self.D.train()

        edge, content_ref, color_ref, img_idx = input
        # convert list of one_d histogram into a tensor of multi-channel histogram
        histogram = torch.stack(
            self.histogram_loss.extract_hist(color_ref, one_d=True, normalize=True), 1,
        )
        ############################
        # D loss
        ############################
        self.optimizer_D.zero_grad()

        gen = self.G(edge, histogram)
        # real y and x -> 1
        loss_D_real = self.gan_loss(self.D(content_ref, edge), 1) * self.lambd_d
        # gen and x -> 0
        loss_D_fake = self.gan_loss(self.D(gen.detach(), edge), 0) * self.lambd_d
        # Combine
        loss_D = loss_D_real + loss_D_fake

        self.d_update(loss_D, i)
        # loss_D.backward()
        # self.optimizer_D.step()

        # self.save_image((x, gen, y), 'datasets/maps/samples', '2018')
        ############################
        # G loss
        ############################
        self.optimizer_G.zero_grad()

        # gen = self.G(x)
        # GAN loss of G
        loss_G_gan = self.gan_loss(self.D(gen, edge), 1)
        # L1 loss of G
        #         loss_G_L1 = self.L1_loss_fn(gen, y) * self.lambd

        # histogram loss
        emd_loss, mi_loss = self.histogram_loss(color_ref, gen)

        # Combine
        loss_G = loss_G_gan + (self.lambda_emd * emd_loss) + (self.lambda_mi * mi_loss)

        loss_G.backward()
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
            "G": loss_G,
            "G_gan": loss_G_gan,
            "G_EMD": emd_loss,
            "G_MI": mi_loss,
            "D": loss_D,
            "D_real": loss_D_real,
            "D_fake": loss_D_fake,
        }

    def eval(self, input, save, out_dir_img, epoch):
        # self.G.eval()
        # self.D.eval()

        with torch.no_grad():
            edge, content_ref, color_ref, img_idx = input
            # convert list of one_d histogram into a tensor of multi-channel histogram
            histogram = torch.stack(
                self.histogram_loss.extract_hist(color_ref, one_d=True, normalize=True),
                1,
            )
            gen = self.G(edge, histogram)

            # self.save_image((x, gen, y), 'datasets/maps/samples', '2018')

            ############################
            # D loss
            ############################
            # real y and x -> 1
            loss_D_real = self.gan_loss(self.D(content_ref, edge), 1) * self.lambd_d
            # gen and x -> 0
            loss_D_fake = self.gan_loss(self.D(gen, edge), 0) * self.lambd_d
            # Combine
            loss_D = loss_D_real + loss_D_fake

            ############################
            # G loss
            ############################
            # GAN loss of G
            loss_G_gan = self.gan_loss(self.D(gen, edge), 1)

            # histogram loss
            emd_loss, mi_loss = self.histogram_loss(color_ref, gen)

            # Combine
            loss_G = (
                loss_G_gan + (self.lambda_emd * emd_loss) + (self.lambda_mi * mi_loss)
            )

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
            "G": loss_G,
            "G_gan": loss_G_gan,
            "G_EMD": emd_loss,
            "G_MI": mi_loss,
            "D": loss_D,
            "D_real": loss_D_real,
            "D_fake": loss_D_fake,
        }

    def test(self, images, i, out_dir_img):
        with torch.no_grad():
            edge, content_ref, color_ref, img_idx = images
            # convert list of one_d histogram into a tensor of multi-channel histogram
            histogram = torch.stack(
                self.histogram_loss.extract_hist(color_ref, one_d=True, normalize=True),
                1,
            )
            gen = self.G(edge, histogram)
            score_gen = self.D(gen, edge).mean()
            score_gt = self.D(content_ref, edge).mean()
            self.save_image(
                (edge, content_ref, content_ref, gen),
                out_dir_img,
                "test_%d" % img_idx,
                test=True,
            )
        return score_gen, score_gt

    def gan_loss(self, out, label):
        return self.gan_loss_fn(
            out, torch.ones_like(out) if label else torch.zeros_like(out)
        )

    def load_state(self, state, lr=None):
        print("Using pretrained model...")
        self.G.load_state_dict(state["G"])
        self.D.load_state_dict(state["D"])
        self.optimizer_G.load_state_dict(state["optimG"])
        self.optimizer_D.load_state_dict(state["optimD"])

        # set model lr to new lr
        if lr is not None:
            for param_group in self.optimizer_G.param_groups:
                before = param_group["lr"]
                param_group["lr"] = lr
            for param_group in self.optimizer_D.param_groups:
                before = param_group["lr"]
                param_group["lr"] = lr
            print("optim lr: before={} / after={}".format(before, lr))

    def save_state(self):
        return {
            "G": self.G.state_dict(),
            "D": self.D.state_dict(),
            "optimG": self.optimizer_G.state_dict(),
            "optimD": self.optimizer_D.state_dict(),
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
