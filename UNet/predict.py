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


boneage_training_dir = "../data/BoneAge/Bone Age Validation Set/boneage-validation-dataset-1/"
boneage_target_dir = "boneage-masks/val1/"


def preprocess(img, size=512):
    # Scale the image according to the given image size
    resize = transforms.Resize((size, size))
    img = resize(img)

    img = transforms.ToTensor()(img)[0].unsqueeze(dim=0)

    img = (img-img.min()) / (img.max()-img.min())
    # plt.imshow(img.permute(1,2,0))
    # plt.show()

    return img


def predict(model):
    to_mask_dirs = ["healthy_magd", "HyCh", "PsHPT", "uts_leipzig", "uts_magdeburg"]
    to_mask_base_dir = "../data/xcat/"
    #to_mask_dirs = ["bone_age_training_data_set"]
    #to_mask_base_dir = "../data/xcat/rsna_bone_age/"

    model.eval()

    def get_img_names(dir):
        filenames = [splitext(file)[0] for file in listdir(dir) if not file.startswith('.')]
        return filenames

    def load_img(img_name, path):
        img_file = glob(f"{path}/{img_name}.*")

        print(path)
        print(img_name)
        print(img_file)

        img = Image.open(img_file[0])

        # plt.imshow(img)
        # plt.show()

        return img

    # Loop through each dir
    for dir in to_mask_dirs:
        dir_path = to_mask_base_dir + dir
        target_dir = f"{to_mask_base_dir}output/{dir}/"

        os.makedirs(target_dir, exist_ok=True)

        img_names = get_img_names(dir_path)
        for img_name in img_names:
            img = load_img(img_name, dir_path)
            original_img_size = img.size #WxH
            img_p = preprocess(img, 512).unsqueeze(0).to(device)

            pred = None
            with torch.no_grad():
                pred = model(img_p).squeeze()

            # Binarize
            #pred = (torch.sigmoid(pred) > 0.5).float()
            pred = torch.sigmoid(pred).float()
            
            # plt.imshow(pred.cpu())
            # plt.show()

            # Save prediction
            transforms.ToPILImage()(pred).save(f"{target_dir}{img_name}.png")

            # Save prediction, resized to original image size
            save_img = transforms.Resize((original_img_size[1], original_img_size[0]))(transforms.ToPILImage()(pred))
            save_img.save(f"{target_dir}{img_name}_resize.png")


def post_process(img):
    #Remove everything except for the largest component, using connected components
    img = (img.squeeze()).numpy()
    img = img.astype(np.uint8)
    nb_components, output, stats,_ = cv2.connectedComponentsWithStats(img, connectivity=8) #might test with 4..

    #Remove the background as a component
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    out_img = np.zeros((img.shape), dtype=np.float32)
    for i in range(0, nb_components):
        if sizes[i] >= max(sizes):
            out_img[output == i+1] = 1.

    # Fill holes
    out_img = ndimage.binary_fill_holes(out_img).astype(np.float32)
    return torch.from_numpy(out_img)


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

    parser.add_argument('--model-type', default='UNet', dest='model_type',
                        help='Model type to use. (Options: UNet, EffUNet)')
    parser.add_argument('--act-type', default='ReLU', dest='act_type',
                        help='activation function to use in UNet. (Options: ReLU, PReLU)')
    parser.add_argument('--up-type', default='upsample', dest='up_type',
                        help='Upsampling type to use in UpConv part of UNet (Options: upsample, upconv)')
    parser.add_argument('--norm', default='BN', dest='norm',
                        help='Which Normalization to use: None, BN, GN')

    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='random seed (default: 11)')

    return parser.parse_args()


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    in_channels = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

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
                     up_type=args.up_type).to(device)
    elif args.model_type == 'EffUNet':
        base = 'efficientnet-b0'
        model = EffUNet(in_channels=in_channels, act_type=act_type, norm_type=nn.GroupNorm, up_type=args.up_type, base=base).to(device)
    else:
        print(f"No valid model type given! (got model_type: {args.model_type})")
        raise NotImplementedError()


    logging.info("Loading model {}".format(args.model))
    model.load_state_dict(torch.load(f"saved_models/s15_bonemask_adam_augment_EffUNet_e50_PReLU_upsample_GN_bs8_scale100_dropout0.0.pt", map_location=device)) # Best performance
    logging.info("Model loaded !")

    predict(model)
