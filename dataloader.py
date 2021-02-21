import os
import random

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class GANDataset(Dataset):

    # Initial logic here, including reading the image files and transform the data
    def __init__(self, args, root_AB, device=None, test=False):
        # initialize image path and transformation
        sorted_AB = sorted(os.listdir(root_AB), key=lambda name: int(name.split("_")[0]))
        self.image_pathsAB = list(map(lambda x: os.path.join(root_AB, x), sorted_AB))

        self.device = device
        self.test = test
        self.opt = args
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    args.crop, Image.BICUBIC
                ),  # resize to crop size directly
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    # override to support indexing
    def __getitem__(self, index):

        AB_path = self.image_pathsAB[index]
        AB = cv2.imread(AB_path)
        AB = cv2.cvtColor(AB, cv2.COLOR_BGR2RGB)

        # split AB image into A and B
        h, w, _ = AB.shape
        w2 = int(w / 3)
        imageA = AB[0:h, 0:w2]
        imageB = AB[0:h, w2 : 2 * w2]
        # imageA = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
        # imageA = np.tile(imageA, (1, 1, 3))
        imageC = AB[0:h, 2 * w2 : w]

        # transform the images if needed
        if self.test:
            imageA = self.transform(imageA)
            imageB = self.transform(imageB)
            imageC = self.transform(imageC)
        else:
            transform_params = get_params(self.opt, imageA.shape[:-1])
            A_transform = get_transform(self.opt, self.mean, self.std, transform_params)
            B_transform = get_transform(self.opt, self.mean, self.std, transform_params)
            C_transform = get_transform(self.opt, self.mean, self.std, transform_params)

            imageA = A_transform(imageA)
            imageB = B_transform(imageB)
            imageC = C_transform(imageC)

        # convert to GPU tensor
        if self.device is not None:
            imageA = imageA.to(self.device)
            imageB = imageB.to(self.device)
            imageC = imageC.to(self.device)

        return imageA, imageB, imageC, index + 1

    # returns the number of examples we read
    def __len__(self):
        # print(len(self.image_pathsA))
        return len(self.image_pathsAB)  # of how many examples we have


# return - DataLoader
def get_dataloader(
    args, image_pathAB, batch_size, resize, crop, device=None, shuffle=True, test=False
):
    batch_dataset = GANDataset(args, image_pathAB, device, test)

    return DataLoader(dataset=batch_dataset, batch_size=batch_size, shuffle=shuffle)


# code below is adapted from
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/base_dataset.py


def get_params(opt, size):
    h, w = size
    new_h = h
    new_w = w
    if opt.preprocess == "resize_and_crop":
        new_h = new_w = opt.resize
    elif opt.preprocess == "scale_width_and_crop":
        new_w = opt.resize
        new_h = opt.resize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop))
    y = random.randint(0, np.maximum(0, new_h - opt.crop))

    flip = random.random() > 0.5

    return {"crop_pos": (x, y), "flip": flip}


def get_transform(
    opt, mean, std, params=None, grayscale=False, method=Image.BICUBIC, convert=True
):
    transform_list = [transforms.ToPILImage()]
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if "resize" in opt.preprocess:
        osize = [opt.resize, opt.resize]
        transform_list.append(transforms.Resize(osize, method))
    elif "scale_width" in opt.preprocess:
        transform_list.append(
            transforms.Lambda(
                lambda img: __scale_width(img, opt.resize, opt.crop, method)
            )
        )

    if "crop" in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop))
        else:
            transform_list.append(
                transforms.Lambda(lambda img: __crop(img, params["crop_pos"], opt.crop))
            )

    if opt.preprocess == "none":
        transform_list.append(
            transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method))
        )

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params["flip"]:
            transform_list.append(
                transforms.Lambda(lambda img: __flip(img, params["flip"]))
            )

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((mean[0],), (std[0],))]
        else:
            transform_list += [transforms.Normalize(mean, std)]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, "has_printed"):
        print(
            "The image size needs to be a multiple of 4. "
            "The loaded image size was (%d, %d), so it was adjusted to "
            "(%d, %d). This adjustment will be done to all images "
            "whose sizes are not multiples of 4" % (ow, oh, w, h)
        )
        __print_size_warning.has_printed = True
