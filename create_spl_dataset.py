import argparse
import glob
import os
import random

from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dir", required=True, type=str)
    parser.add_argument("--new_dir", required=True, type=str)
    parser.add_argument("--n_repeat", required=True, type=str)
    return parser.parse_args()


def main(args):
    original_dir = args.original_dir
    assert os.path.exists(original_dir), f"{original_dir} does not exist"
    new_dir = args.new_dir
    dataset_dir = os.listdir(original_dir)
    for folder in dataset_dir:
        print(f"Creating new images in {new_dir}/{folder} from {original_dir}/{folder}")
        folder_images = glob.glob(f"{original_dir}/{folder}/*")
        out_dir = os.path.join(new_dir, folder)
        os.makedirs(out_dir, exist_ok=True)

        for img in folder_images:
            img_filename = os.path.split(img)[-1].split(".")[0]
            img = Image.open(img)
            random_picks = random.choices(folder_images, k=args.n_repeat)
            for i, img2 in enumerate(random_picks):
                new_img = Image.new("RGB", (256 * 3, 256))
                img2 = Image.open(img2)
                img2 = img2.crop((256, 0, 512, 256))
                new_img.paste(img, (0, 0))
                new_img.paste(img2, (512, 0))
                new_img.save(os.path.join(out_dir, f"{img_filename}_{i}.jpg"))


if __name__ == "__main__":
    args = get_args()
    main(args)
