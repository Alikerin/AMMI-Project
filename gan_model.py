import itertools
import os

import imageio
import numpy as np
import torch
from torch.nn import init

from histogram import HistLayer, extract_1d_hist, extract_hist, histogram_losses
from model_modules import Discriminator, Generator


class GANModel:
    def __init__(self, args):
        self.start_epoch = 0
        self.args = args

        self.G = Generator()
        self.D = Discriminator()
        self.hist_layer = HistLayer(in_channels=1, num_bins=256)

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
        self.lambda_emd = args.lambda_emd
        self.lambda_mi = args.lambda_mi

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
        self.hist_layer.to(device)

        for state in itertools.chain(
            self.optimizer_G.state.values(), self.optimizer_D.state.values()
        ):
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def train(self, input, save, out_dir_img, epoch, i):
        # self.G.train()
        # self.D.train()

        edge, img, img_idx = input
        histogram = None  # extract_1d_hist(self.hist_layer, img)

        ############################
        # D loss
        ############################
        self.optimizer_D.zero_grad()

        gen = self.G(edge, histogram)
        # real y and x -> 1
        loss_D_real = self.gan_loss(self.D(img, edge), 1) * self.lambd_d
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
        # h_real = extract_hist(self.hist_layer, img)
        # h_gen = extract_hist(self.hist_layer, gen)
        # emd_loss, mi_loss = histogram_losses(h_real, h_gen)
        loss_G_EMD = 0  # self.lambda_emd * emd_loss
        loss_G_MI = 0  # self.lambda_mi * mi_loss

        # Combine
        loss_G = loss_G_gan  # + loss_G_MI + loss_G_EMD

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
            "G_MI": loss_G_MI,
            "G_EMD": loss_G_EMD,
            "D": loss_D,
            "D_real": loss_D_real,
            "D_fake": loss_D_fake,
        }

    def eval(self, input, save, out_dir_img, epoch):
        # self.G.eval()
        # self.D.eval()

        with torch.no_grad():
            edge, img, img_idx = input
            histogram = None  # extract_1d_hist(self.hist_layer, img)
            gen = self.G(edge, histogram)

            # self.save_image((x, gen, y), 'datasets/maps/samples', '2018')

            ############################
            # D loss
            ############################
            # real y and x -> 1
            loss_D_real = self.gan_loss(self.D(img, edge), 1) * self.lambd_d
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
            # h_real = extract_hist(self.hist_layer, img)
            # h_gen = extract_hist(self.hist_layer, gen)
            # emd_loss, mi_loss = histogram_losses(h_real, h_gen)
            loss_G_EMD = 0  # self.lambda_emd * emd_loss
            loss_G_MI = 0  # self.lambda_mi * mi_loss

            # Combine
            loss_G = loss_G_gan  # + loss_G_MI + loss_G_EMD

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
            "G_MI": loss_G_MI,
            "G_EMD": loss_G_EMD,
            "D": loss_D,
            "D_real": loss_D_real,
            "D_fake": loss_D_fake,
        }

    def test(self, images, i, out_dir_img):
        with torch.no_grad():
            A, B, img_idx = images
            histogram = extract_1d_hist(self.hist_layer, B)
            gen = self.G(A, histogram)
            score_gen = self.D(gen, A).mean()
            score_gt = self.D(B, A).mean()
            self.save_image((A, B, gen), out_dir_img, "test_%d" % img_idx, test=True)
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
        A, B, gen = input

        if test:
            img = self.tensor2image(gen)
            path = os.path.join(filepath, "%s.png" % fname)
            imageio.imwrite(path, img.squeeze().transpose(1, 2, 0))
        else:
            merged = self.tensor2image(self.merge_images(A, B, gen))
            path = os.path.join(filepath, "%s.png" % fname)
            imageio.imwrite(path, merged)

        print("saved %s" % path)

    def tensor2image(self, input):
        image_data = input.data
        image = 127.5 * (image_data.cpu().float().numpy() + 1.0)
        return image.astype(np.uint8)

    def merge_images(self, sources, targets, generated):
        # row, _, h, w = sources.shape
        row, _, h, w = sources.size()
        # row = int(np.sqrt(batch_size))
        # merged = np.zeros([3, row * h, w * 3])
        merged = torch.zeros([3, row * h, w * 3])
        for idx, (s, t, g) in enumerate(zip(sources, targets, generated)):
            i = idx
            # i = (idx + 1) // row
            # j = idx % row
            # merged[:, i * h:(i + 1) * h, (j * 2) * w:(j * 2 + 1) * w] = s
            # merged[:, i * h:(i + 1) * h, (j*2+1) * w:(j * 2 + 2) * w] = t
            # merged[:, i * h:(i + 1) * h, (j*2+2) * w:(j * 2 + 3) * w] = c
            merged[:, i * h : (i + 1) * h, 0:w] = s
            merged[:, i * h : (i + 1) * h, w : 2 * w] = g
            merged[:, i * h : (i + 1) * h, 2 * w : 3 * w] = t
        return merged.permute(1, 2, 0)
