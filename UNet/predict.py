import argparse
import csv
import logging
import os
from glob import glob
from os.path import splitext
from os import listdir

import cv2
from PIL import Image
from torchvision import transforms
from scipy import ndimage

from lib.datasets.bonemask_dataset import BoneMaskDataset
from lib.models.EffUNet import EffUNet
from lib.models.UNet import UNet
from lib.utils import *


import cv2
import torch
import os
from glob import glob
import matplotlib.pyplot as plt

from lib.models import *
from lib.datasets import *
from tqdm import tqdm

from argparse import ArgumentParser

class Predictor:
    def __init__(
        self,
        model,
        size=512,
        use_gpu=False,
    ):
        self.model = model
        self.model.eval()
        if use_gpu:
            self.model.cuda()
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.size = size

    def __call__(self, img_path: str) -> np.array:
        """
        provide a path to an image and the predictor return the extracted hand contour
        """
        img = self.load_img(img_path)
        original_img_size = img.size  # WxH
        img_p = self.preprocess(img, 512).unsqueeze(0).to(self.device)

        pred = None
        with torch.no_grad():
            pred = self.model(img_p).squeeze()

        pred = torch.sigmoid(pred).float()

        # Save prediction, resized to original image size
        save_img = np.array(transforms.Resize((original_img_size[1], original_img_size[0]))(
            transforms.ToPILImage()(pred)))

        conts, _ = cv2.findContours(save_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = max(conts, key=cv2.contourArea)

        image_crop = cv2.cvtColor((np.array(img) // 256).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        vis = cv2.drawContours(image_crop, [c], -1, (0, 255, 0), 5)
        return save_img, vis

    @staticmethod
    def load_img(img_file):
        img = Image.open(img_file)
        return img

    @staticmethod
    def preprocess(img, size=512):
        # Scale the image according to the given image size
        resize = transforms.Resize((size, size))
        img = resize(img)
        img = transforms.ToTensor()(img)[0].unsqueeze(dim=0)
        img = (img - img.min()) / (img.max() - img.min())
        return img


def main(
    input,
    model,
    size=512,
    output="./output/",
    use_gpu=False,
):
    p = Predictor(model, size, use_gpu=use_gpu)
    os.makedirs(output, exist_ok=True)
    if os.path.isdir(input):
        for img_path in tqdm(
            glob(os.path.join(input, "*.png"))
            + glob(os.path.join(input, "*.jpg"))
        ):
            hand, vis = p(img_path)
            cv2.imwrite(os.path.join(output, os.path.basename(img_path)), hand)
            cv2.imwrite(
                os.path.join(
                    output, os.path.basename(img_path).replace(".png", "_vis.png")
                ),
                vis,
            )
    else:
        cv2.imwrite(os.path.join(output, os.path.basename(input)), p(input))


# TODO: Some of these are deprecated and should be removed or re-implemented
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--output', '-o', default='examples_output/',
                        help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument("--model", default="/home/rassman/bone2gene/hand-segmentation/UNet/lib/output/weights/s15_bonemask_adam_augment_EffUNet_e50_PReLU_upsample_GN_bs8_scale100_dropout0.0.pt")

    parser.add_argument('--model-type', default='EffUNet', dest='model_type',
                        help='Model type to use. (Options: UNet, EffUNet)')
    parser.add_argument('--act-type', default='PReLU', dest='act_type',
                        help='activation function to use in UNet. (Options: ReLU, PReLU)')
    parser.add_argument('--up-type', default='upsample', dest='up_type',
                        help='Upsampling type to use in UpConv part of UNet (Options: upsample, upconv)')
    parser.add_argument('--norm', default='GN', dest='norm',
                        help='Which Normalization to use: None, BN, GN')

    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='random seed (default: 11)')

    parser.add_argument(
        "--checkpoint",
        default="/home/rassman/bone2gene/masking/output/version_25/ckp/best_model.ckpt",
    )
    parser.add_argument(
        "--input",
        default="/home/rassman/bone2gene/data/annotated/shox_magdeburg/shox_magd_00001.png",
        help="can be eiter single image or full directory",
    )
    parser.add_argument(
        "--input_size", default=512, type=int,
    )
    parser.add_argument(
        "--output_dir", default="./output/masks/",
    )
    parser.add_argument("--use_gpu", action="store_true")

    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    in_channels = 1

    if args.norm == 'BN':
        norm_type = nn.BatchNorm2d
    elif args.norm == 'LN':
        norm_type = nn.LayerNorm
    elif args.norm == 'IN':
        norm_type = nn.InstanceNorm2d
    elif args.norm == 'GN':
        norm_type = nn.GroupNorm
    elif args.norm == 'None':
        norm_type = None
    else:
        print(f"Unknown Normalization type given: {args.norm}")
        raise NotImplementedError()
    print(f"Using Normalization type: {norm_type}")

    if args.act_type == "ReLU":
        act_type = nn.ReLU
    elif args.act_type == "PReLU":
        act_type = nn.PReLU
    else:
        print(f"Invalid act_type given! (Got {args.act_type})")
        raise NotImplementedError()

    if args.model_type == 'UNet':
        model = UNet(depth=5, in_channels=in_channels, num_classes=1, padding=1, act_type=act_type, norm_type=norm_type,
                     up_type=args.up_type)
    elif args.model_type == 'EffUNet':
        base = 'efficientnet-b0'
        model = EffUNet.EffUNet(in_channels=in_channels, act_type=act_type, norm_type=nn.GroupNorm, up_type=args.up_type, base=base)
    else:
        print(f"No valid model type given! (got model_type: {args.model_type})")
        raise NotImplementedError()


    logging.info("Loading model {}".format(args.model))
    model.load_state_dict(torch.load(args.model))
    logging.info("Model loaded !")

    main(args.input, model, args.input_size, args.output_dir, args.use_gpu)
