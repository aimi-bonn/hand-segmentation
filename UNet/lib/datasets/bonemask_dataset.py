from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF

import random

from torchvision import transforms
import matplotlib.pyplot as plt


##
# Implementation by: milesial
# https://github.com/milesial/Pytorch-UNet
##

class BoneMaskDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir=-1, size=512, mask_suffix='_mask', img_suffix=-1, validation=False):
        self.imgs_dir = imgs_dir

        if masks_dir == -1:
            self.masks_dir = imgs_dir
        else:
            self.masks_dir = masks_dir

        self.size = size
        self.mask_suffix = mask_suffix
        self.img_suffix = img_suffix

        self.ids = [(splitext(file)[0]).split('_')[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        self.validation = validation
        if validation:
            self.augment_transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.augment_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.75, contrast=0.25, saturation=0.5),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.ids)

    # TODO: Replace all random.random() with either np or torch, as random.random() is NOT reproducable!
    def preprocess(self, img, mask, augment):
        img = np.array(img)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = Image.fromarray(np.uint8(img))

        if not self.validation:
            w, h = img.size
            # Add color gradient edge to image
            if random.random() > 0.5:
                line_thickness = random.random() * 0.03  # Max 5% of image width or height

                def interpolate(f_co, t_co, interval):
                    det_co = [(t - f) / interval for f, t in zip(f_co, t_co)]
                    for i in range(interval):
                        yield [round(f + det * i) for f, det in zip(f_co, det_co)]

                gradient = Image.new('RGBA', img.size, color=0)
                img_draw = ImageDraw.Draw(gradient)

                # Horizontal edge
                if random.random() > 0.5:
                    from_top = True if random.random() > 0.5 else False
                    #line_thickness = int(line_thickness * h + 0.02 * h)
                    for i, color in enumerate(interpolate((255, 255, 255), (0, 0, 0), int(line_thickness * h + 0.02 * h))):
                        color.append(127)
                        img_draw.line(((0, i if from_top else h-i), (w, i if from_top else h-i)), tuple(color), width=1)

                # Vertical edge
                if random.random() > 0.5:
                    from_left = True if random.random() > 0.5 else False
                    #line_thickness = int(line_thickness * w + 0.02 * w)
                    for i, color in enumerate(interpolate((255, 255, 255), (0, 0, 0), int(line_thickness * w + 0.02 * w))):
                        color.append(127)
                        img_draw.line(((i if from_left else w-i, 0), (i if from_left else w-i, h)), tuple(color), width=1)

                img = Image.alpha_composite(img, gradient)


            # Horizontally flip the image
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            # Vertically flip the image
            if random.random() > 0.85:  # Small chance as we don't really see it in the dataset
                img = TF.vflip(img)
                mask = TF.vflip(mask)

            # Add a random gray patch to the image
            intensity = random.random()
            if intensity > 0.5:
                rect = Image.new('RGBA', img.size)
                img_draw = ImageDraw.Draw(rect, "RGBA")
                x,y = random.random()*w, random.random()*h
                x2, y2 = random.random()*w, random.random()*h
                intensity = int(200*intensity)
                img_draw.rectangle(((x, y), (x2, y2)), fill=(intensity, intensity, intensity, 127))
                img = Image.alpha_composite(img, rect)
                # plt.imshow(img)
                # plt.show()

            # Overlay the entire image with a high intensity rectangle
            intensity = random.random()
            if intensity > 0.80:
                rect = Image.new('RGBA', img.size)
                img_draw = ImageDraw.Draw(rect, "RGBA")
                x,y = 0,0
                x2, y2 = img.size
                intensity = int(200*intensity)
                img_draw.rectangle(((x, y), (x2, y2)), fill=(intensity, intensity, intensity, 127))
                img = Image.alpha_composite(img, rect)
            # Adjust image gamma to brighten aspects of the image
            elif random.random() > 0.75:
                gamma = random.random() * 0.5 + 0.4 # in range [0.4, 0.9]
                img = transforms.functional.adjust_gamma(img, gamma)

            ## From this point onwards the image dimensions may change!
            # Rotate the image [-20, 20] degrees
            r_color = -1
            if random.random() > 0.3:
                rot = (random.random()-0.5) * 2 * 20.0
                # Use a white or black background?
                if random.random() > 0.66:
                    r_color = 0
                    fill_color = tuple([0]) if 'L' in img.getbands() else tuple([0,0,0])
                elif random.random() > 0.66:
                    r_color = 255
                    fill_color = tuple([255]) if 'L' in img.getbands() else tuple([255,255,255])
                else:
                    r_color = int(random.random()*255)
                    fill_color = tuple([r_color]) if 'L' in img.getbands() else tuple([r_color,r_color,r_color])

                img = img.rotate(angle=rot, resample=False, expand=True,
                                 fillcolor=fill_color)
                mask = mask.rotate(angle=rot, resample=False, expand=True, fillcolor=(0, 0, 0))

            # Pad around image to augment similar to SHOX
            if random.random() > 0.8:
                # Use a white or black background?
                if random.random() > 0.66:
                    fill_color = tuple([0]) if 'L' in img.getbands() else tuple([0,0,0,255])
                elif random.random() > 0.66:
                    fill_color = tuple([255]) if 'L' in img.getbands() else tuple([255,255,255,255])
                else:
                    # we want to use the same background color as the rotational padding, if there was any
                    if r_color == -1:
                        r_color = int(random.random()*105) + 150 # range 150-255
                    fill_color = tuple([r_color]) if 'L' in img.getbands() else tuple([r_color,r_color,r_color,255])

                # Amount of padding to add to the borders: max 25% of width or height on each side
                x1_pad, y1_pad, x2_pad, y2_pad = int(random.random()*0.25*w), int(random.random()*0.25*h),\
                                                 int(random.random()*0.25*w), int(random.random()*0.25*h)

                img = transforms.functional.pad(img, padding=(x1_pad, y1_pad, x2_pad, y2_pad), fill=fill_color)
                mask = transforms.functional.pad(mask, padding=(x1_pad, y1_pad, x2_pad, y2_pad), fill=(0,0,0,255))



        # Scale the image according to the given image size
        resize = transforms.Resize((self.size, self.size))
        img = resize(img)
        
        #TODO: Should never happen, can probably be removed..
        if img.size != mask.size:
            mask = resize(mask)
            if img.size != mask.size:
                print("After multiple attempts, mask and input are not the same size!")
                exit()

        if augment:
            img = self.augment_transform(img)[0].unsqueeze(dim=0)

        mask = transforms.ToTensor()(mask)[0].unsqueeze(dim=0)

        return img, mask

    # TODO: replace glob(..) to a implementation similar to that in the boneage dataset I made .. this seems to take a rediculous amount of time
    def __getitem__(self, i, to_augment=True):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + self.img_suffix + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        img = img.convert('RGBA')
        img, mask = self.preprocess(img, mask, to_augment)

        return img, mask
