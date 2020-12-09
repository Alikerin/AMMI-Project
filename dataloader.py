import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class GANDataset(Dataset):

    # Initial logic here, including reading the image files and transform the data
    def __init__(self, root_AB, transform=None, device=None, test=False):
        # initialize image path and transformation
        sorted_AB = sorted(
            os.listdir(root_AB), key=lambda name: int(name.split("_")[0])
        )
        self.image_pathsAB = list(map(lambda x: os.path.join(root_AB, x), sorted_AB))

        self.transform = transform
        self.device = device
        self.test = test

    # override to support indexing
    def __getitem__(self, index):

        AB_path = self.image_pathsAB[index]
        AB = Image.open(AB_path).convert("RGB")
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        imageA = AB.crop((0, 0, w2, h))
        imageB = AB.crop((w2, 0, w, h))

        # transform the images if needed
        if self.transform is not None:
            if self.test:
                imageA = self.transform(imageA)
                imageB = self.transform(imageB)
            else:  # if test is False, need to crop and flip the same way
                # setting the same seed for input and target tansformations
                seed = np.random.randint(2147483647)
                random.seed(seed)
                imageA = self.transform(imageA)
                random.seed(seed)
                imageB = self.transform(imageB)

        # convert to GPU tensor
        if self.device is not None:
            imageA = imageA.to(self.device)
            imageB = imageB.to(self.device)

        return imageA, imageB, index + 1

    # returns the number of examples we read
    def __len__(self):
        # print(len(self.image_pathsA))
        return len(self.image_pathsAB)  # of how many examples we have


## return - DataLoader
def get_dataloader(
    image_pathAB, batch_size, resize, crop, device=None, shuffle=True, test=False
):
    if test:
        transform = transforms.Compose(
            [
                transforms.Resize(crop, Image.BICUBIC),  # resize to crop size directly
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                # resize PIL image to given size
                transforms.Resize(resize, Image.BICUBIC),
                # crop image at randomn location
                transforms.RandomCrop(crop),
                # flip images randomly
                transforms.RandomHorizontalFlip(),
                # convert image input into torch tensor
                transforms.ToTensor(),
                # normalize image with mean and standard deviation
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    batch_dataset = GANDataset(image_pathAB, transform, device, test)

    return DataLoader(dataset=batch_dataset, batch_size=batch_size, shuffle=shuffle)
