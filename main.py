import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from visdom import Visdom

import dataloader
from gan_model import GANModel

parser = argparse.ArgumentParser()
# Model
parser.add_argument("--unaligned", default=False, type=bool)
parser.add_argument("--resize", default=286, type=int)
parser.add_argument("--crop", default=256, type=int)
parser.add_argument(
    "--no_flip",
    action="store_true",
    help="if specified, do not flip the images for data augmentation",
)
parser.add_argument(
    "--preprocess",
    type=str,
    default="resize_and_crop",
    help="scaling and cropping of images at load time [resize_and_crop | crop | \
        scale_width | scale_width_and_crop | none]",
)
parser.add_argument(
    "--color_space",
    type=str,
    default="RBG",
    help="color space for histogram extraction [RGB | YCbCr | LAB]",
)
# Training
parser.add_argument("--device_id", default=0, type=int)
parser.add_argument("--mode", default="train", type=str)
parser.add_argument("--pretrain_path", default="", type=str)
parser.add_argument("--print_every_train", default=100, type=int)
parser.add_argument("--print_every_val", default=200, type=int)
parser.add_argument("--save_every_epoch", default=20, type=int)
parser.add_argument(
    "--eval_n",
    default=100,
    type=int,
    help="number of examples from val set to evaluate on each epoch",
)
parser.add_argument(
    "--save_n_img",
    default=10000,
    type=int,
    help="number of images to save at test time",
)
parser.add_argument("--suffix", default="", type=str, help="out dir suffix")
# Optimization
parser.add_argument("--lr", default=0.0002, type=float)
parser.add_argument(
    "--lr_decay_start", default=100, type=int, help="eppch to start lr decay"
)
parser.add_argument(
    "--lr_decay_n", default=100, type=int, help="number of epochs to decay lr to 0"
)
parser.add_argument("--wd", default=0, type=float)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--dropout", default=0.5, type=float)
parser.add_argument("--bias", default=False, type=bool)
parser.add_argument("--norm", default="batch", type=str, help="batch|instance|none")
parser.add_argument(
    "--G", default="unet", type=str, help="unet|resnet6|resnet9|resnet50|resnet101"
)
parser.add_argument("--D", default="patch", type=str, help="patch|image")
parser.add_argument("--gan_loss", default="BCE", type=str, help="BCE|MSE")
parser.add_argument("--n_epoch", default=100, type=int)
parser.add_argument("--beta1", default=0.5, type=float, help="momentum term of adam")
parser.add_argument("--lambd", default=100.0, type=float, help="weight for L1 loss")
parser.add_argument("--lambd_d", default=0.5, type=float, help="D loss scale")
parser.add_argument("--lambda_emd", default=0.5, type=float, help="EMD Loss Scale")
parser.add_argument("--lambda_mi", default=0.5, type=float, help="MI loss scale")
parser.add_argument(
    "--color_ref", default="", type=str, help="Color reference image path"
)
parser.add_argument(
    "--d_update_frequency",
    default=1,
    type=int,
    help="discriminator parameter update frequency",
)
parser.add_argument(
    "--init_type",
    default="normal",
    type=str,
    help="initialization for weights for G and D. normal|xavier|kaiming",
)
# Files
parser.add_argument("--out_dir", default="./checkpoints", type=str)
parser.add_argument("--data_dir", default="./datasets/maps/", type=str)

# Visualization
parser.add_argument("--vis", default=False, action="store_true")
parser.add_argument("--port", default=8097, type=int)


if __name__ == "__main__":

    args = parser.parse_args()
    device = torch.device(
        "cuda:%d" % args.device_id if torch.cuda.is_available() else "cpu"
    )
    c_transform = transforms.Compose(
        [
            transforms.Resize(
                [args.crop, args.crop], Image.BICUBIC
            ),  # resize to crop size directly
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    s = "Using %s\n\n" % device
    for k, v in vars(args).items():
        s += "%s = %s\n" % (k, v)
    print(s)

    # output files
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    if args.mode == "train":
        out_dir = os.path.join(
            args.out_dir,
            "%s%s"
            % (
                "output_" + time.strftime("%m%d%H%M%S") + "_s",
                "_" + args.suffix if len(args.suffix) != 0 else "",
            ),
        )
        os.mkdir(out_dir)
        out_dir_img = os.path.join(out_dir, "images")
        os.mkdir(out_dir_img)
        log_file = os.path.join(out_dir, "train.json")
        config_file = os.path.join(out_dir, "config.txt")

        # save configs to config file
        with open(config_file, "w") as f:
            f.write(s)
        print("\nSave model and stats to directory %s" % (out_dir))

        # load data
        train_loader = dataloader.get_dataloader(
            args,
            os.path.join(args.data_dir, "train"),
            resize=args.resize,
            crop=args.crop,
            shuffle=True,
            test=False,
            batch_size=args.batch_size,
            device=device,
        )
        val_loader = dataloader.get_dataloader(
            args,
            os.path.join(args.data_dir, "val"),
            resize=args.resize,
            crop=args.crop,
            shuffle=True,
            test=True,
            batch_size=1,
            device=device,
        )  # TODO val batch size
    if args.mode == "test":
        out_dir = os.path.dirname(args.pretrain_path)
        out_dir_img = os.path.join(
            out_dir, "images", "test_" + time.strftime("%m%d%H%M%S") + "_s"
        )
        os.makedirs(out_dir_img, exist_ok=True)

        # load data
        test_loader = dataloader.get_dataloader(
            args,
            os.path.join(args.data_dir, "test"),
            resize=args.resize,
            crop=args.crop,
            shuffle=False,
            test=True,
            batch_size=1,
            device=device,
        )

    if args.vis:
        if args.port:
            viz = Visdom(port=int(args.port))
        else:
            viz = Visdom()

        startup_sec = 1.0
        while not viz.check_connection() and startup_sec > 0:
            time.sleep(0.1)
            startup_sec -= 0.1
        assert viz.check_connection(), "No connection could be formed quickly"

        win_train_G = viz.line(X=np.asarray([0]), Y=np.asarray([0]))
        win_train_D = viz.line(X=np.asarray([0]), Y=np.asarray([0]))
        # win_train_tot = viz.line(X=np.asarray([0]), Y=np.asarray([0]))
        win_eval_G = viz.line(X=np.asarray([0]), Y=np.asarray([0]))
        win_eval_D = viz.line(X=np.asarray([0]), Y=np.asarray([0]))
        # win_eval_tot  = viz.line(X=np.asarray([0]), Y=np.asarray([0]))
        # print('train window id =', win_train)
        # print('eval window id =', win_eval)
    else:
        viz = None

    model = GANModel(args)

    # use pretrain
    start_epoch = 1
    if args.pretrain_path:
        print("\nLoading model from %s, mode: %s" % (args.pretrain_path, args.mode))
        if args.mode == "train":
            # TODO load GPU model on CPU
            checkpoint = torch.load(args.pretrain_path)
            start_epoch = checkpoint["epoch"] + 1
            model.load_state(checkpoint["model_state"])
        if args.mode == "test":
            checkpoint = torch.load(args.pretrain_path, map_location=device)
            model.load_state(checkpoint["model_state"])

    model.set_start_epoch(start_epoch)
    model.to(device)

    if args.mode == "train":
        stats = {}
        stats["train_loss"] = {}
        stats["val_loss"] = {}

        train_vis_iter = 0
        eval_vis_iter = 0
        total_train_iter = len(train_loader)
        eval_n = min(args.eval_n, len(val_loader))  # TODO val batch
        total_val_iter = eval_n

        for epoch in range(start_epoch, start_epoch + args.n_epoch):
            print("\n==== Epoch {:d} ====".format(epoch))
            t_start = time.time()

            # train
            for i, images in enumerate(train_loader):

                loss = model.train(
                    images, save=(i == 0), out_dir_img=out_dir_img, epoch=epoch, i=i
                )

                # update stats
                s = ""
                for k, v in loss.items():
                    if stats["train_loss"].get(k) is None:
                        stats["train_loss"][k] = []
                    # convert Tensor to float
                    v = round(float(v), 4)
                    stats["train_loss"][k].append(v)
                    loss[k] = v
                    s += "%s %f   " % (k, v)

                if i % args.print_every_train == 0:
                    print("Iter %d/%d    loss %s" % (i, total_train_iter, s))

                # visualize train loss
                if viz:
                    viz.line(
                        X=np.asarray([train_vis_iter]),
                        Y=np.asarray([loss["G"]]),
                        name="G",
                        win=win_train_G,
                    )
                    viz.line(
                        X=np.asarray([train_vis_iter]),
                        Y=np.asarray([loss["G_gan"]]),
                        name="G_gan",
                        win=win_train_G,
                    )
                    viz.line(
                        X=np.asarray([train_vis_iter]),
                        Y=np.asarray([loss["G_MI"]]),
                        name="G_MI",
                        win=win_train_G,
                    )
                    viz.line(
                        X=np.asarray([train_vis_iter]),
                        Y=np.asarray([loss["G_EMD"]]),
                        name="G_EMD",
                        win=win_train_G,
                    )

                    viz.line(
                        X=np.asarray([train_vis_iter]),
                        Y=np.asarray([loss["D_fake"]]),
                        name="D_fake",
                        win=win_train_D,
                    )
                    viz.line(
                        X=np.asarray([train_vis_iter]),
                        Y=np.asarray([loss["D_real"]]),
                        name="D_real",
                        win=win_train_D,
                    )
                    viz.line(
                        X=np.asarray([train_vis_iter]),
                        Y=np.asarray([loss["D"]]),
                        name="D",
                        win=win_train_D,
                    )
                train_vis_iter += 1

            print("Time taken: %.2f m" % ((time.time() - t_start) / 60))

            # eval
            if eval_n > 0:
                print("\nEvaluating %d examples on val set..." % eval_n)
                total_val_loss = {}
                for i, images in enumerate(val_loader):
                    if i >= args.eval_n:
                        i -= 1
                        break

                    loss = model.eval(
                        images, save=(i == 0), out_dir_img=out_dir_img, epoch=epoch
                    )

                    # update stats
                    s = ""
                    for k, v in loss.items():
                        if stats["val_loss"].get(k) is None:
                            stats["val_loss"][k] = []
                        if total_val_loss.get(k) is None:
                            total_val_loss[k] = 0
                        # convert Tensor to float
                        v = round(float(v), 4)
                        stats["val_loss"][k].append(v)
                        total_val_loss[k] += v
                        loss[k] = v
                        s += "%s %f   " % (k, v)

                    if i % args.print_every_val == 0:
                        print("Iter %d/%d    loss %s" % (i, total_val_iter, s))

                    # visualize eval loss
                    if viz:
                        viz.line(
                            X=np.asarray([train_vis_iter]),
                            Y=np.asarray([loss["G"]]),
                            name="G",
                            win=win_eval_G,
                        )
                        viz.line(
                            X=np.asarray([train_vis_iter]),
                            Y=np.asarray([loss["G_gan"]]),
                            name="G_gan",
                            win=win_eval_G,
                        )
                        viz.line(
                            X=np.asarray([train_vis_iter]),
                            Y=np.asarray([loss["G_MI"]]),
                            name="G_MI",
                            win=win_eval_G,
                        )
                        viz.line(
                            X=np.asarray([train_vis_iter]),
                            Y=np.asarray([loss["G_EMD"]]),
                            name="G_EMD",
                            win=win_eval_G,
                        )

                        viz.line(
                            X=np.asarray([train_vis_iter]),
                            Y=np.asarray([loss["D_fake"]]),
                            name="D_fake",
                            win=win_eval_D,
                        )
                        viz.line(
                            X=np.asarray([train_vis_iter]),
                            Y=np.asarray([loss["D_real"]]),
                            name="D_real",
                            win=win_eval_D,
                        )
                        viz.line(
                            X=np.asarray([train_vis_iter]),
                            Y=np.asarray([loss["D"]]),
                            name="D",
                            win=win_eval_D,
                        )
                    eval_vis_iter += 1

                # calculate avg val loss
                s = ""
                for k, v in total_val_loss.items():
                    s += "%s %f   " % (k, v / (i + 1))
                print("Average val loss    %s" % s)

            # save stats
            with open(log_file, "w") as f:
                json.dump(stats, f)

            # save model
            if epoch % args.save_every_epoch == 0:
                model_file = os.path.join(out_dir, "epoch_%d.pt" % epoch)
                print("\nSaving model to %s\n" % (model_file))
                torch.save(
                    {"epoch": epoch, "model_state": model.save_state()}, model_file
                )

            # update scheduler
            model.update_scheduler()

        # save model from last epoch
        model_file = os.path.join(out_dir, "epoch_%d.pt" % epoch)
        print("\nSaving model to %s\n" % (model_file))
        torch.save({"epoch": epoch, "model_state": model.save_state()}, model_file)

    if args.mode == "test":
        print("\nEvaluating on test set...")
        scores_gen = []
        scores_gt = []
        for i, images in enumerate(test_loader):
            if i >= args.save_n_img:
                break
            # model.test(images, i, out_dir_img)
            # add color-reference image
            if args.color_ref:
                color_ref_img = Image.open(args.color_ref).convert(args.color_space)
                color_ref_img = c_transform(color_ref_img)
                color_ref_img = color_ref_img.to(device).unsqueeze(0)
                images.append(color_ref_img)
            else:
                images.append(None)
            score_gen, score_gt = model.test(images, i, out_dir_img)
            score_gen = round(float(score_gen), 6)
            score_gt = round(float(score_gt), 6)
            scores_gen.append(score_gen)
            scores_gt.append(score_gt)

            with open(os.path.join(out_dir, "d_scores.json"), "w") as f:
                json.dump({"scores_gen": scores_gen, "scores_gt": scores_gt}, f)

        # test_loss = {}
        # for i, images in enumerate(test_loader):
        #     loss = model.eval(images)
        #
        #     for k, v in loss.items():
        #         if test_loss.get(k) is None:
        #             test_loss[k] = 0
        #         v = round(float(v), 4)
        #         test_loss[k] += v
        #
        # s = ""
        # for k, v in test_loss.items():
        #     test_loss[k] = round(v / (i+1), 4)
        #     s += "%s %f   " % (k, test_loss[k])
        #
        # print("Average loss %s" % (s))
        #
        # log_file = os.path.join(out_dir, "test.json")
        # with open(log_file, "w") as f:
        #     json.dump(test_loss, f)
