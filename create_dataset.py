import argparse
import glob
import os
import random

import cv2
import numpy as np
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dir", required=True, type=str)
    parser.add_argument("--new_dir", required=True, type=str)
    parser.add_argument("--n_repeat", required=True, type=int)
    parser.add_argument("--data_ratio", default=1.0, type=float)
    parser.add_argument("--gray", default=False, action="store_true")
    return parser.parse_args()


def main(args):
    original_dir = args.original_dir
    assert os.path.exists(original_dir), f"{original_dir} does not exist"
    new_dir = args.new_dir
    dataset_dir = os.listdir(original_dir)
    for folder in dataset_dir:
        print(f"Creating new images in {new_dir}/{folder} from {original_dir}/{folder}")
        folder_images = glob.glob(f"{original_dir}/{folder}/*")
        folder_images = folder_images[: int(args.data_ratio * len(folder_images))]
        out_dir = os.path.join(new_dir, folder)
        os.makedirs(out_dir, exist_ok=True)

        for img in folder_images:
            img_filename = os.path.split(img)[-1].split(".")[0]
            AB = Image.open(img)
            w, h = AB.size
            w2 = int(w / 2)
            A = AB.crop((0, 0, w2, h))
            B = AB.crop((w2, 0, w, h))
            if args.gray:
                A = cv2.cvtColor((np.asarray(B)), cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
                A = Image.fromarray(np.tile(A, (1, 1, 3)))
            random_picks = random.choices(folder_images, k=args.n_repeat)
            for i, img2 in enumerate(random_picks):
                new_img = Image.new("RGB", (256 * 3, 256))
                img2 = Image.open(img2)
                w, h = img2.size
                w2 = int(w / 2)
                C = img2.crop((w2, 0, w, h))
                new_img.paste(A, (0, 0))
                new_img.paste(B, (w2, 0))
                new_img.paste(C, (w2 * 2, 0))
                new_img.save(os.path.join(out_dir, f"{img_filename}_{i}.jpg"))


if __name__ == "__main__":
    args = get_args()
    main(args)
