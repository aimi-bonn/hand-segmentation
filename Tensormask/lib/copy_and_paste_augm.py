"""
class offering simple copy and paste data augmentation
 (compare https://arxiv.org/abs/2012.07177).
Optimized to integrate with detectron2
"""
import glob
import itertools
import json
import os
import warnings
from abc import abstractmethod

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

import lib.constants as constants


class PatchCreator:
    """
    class handling cutting out objects as image patches from the image bases the
     corresponding COCO instance annotation file
    """

    frame_tol = 5

    def __init__(
        self,
        coco,
        img_dir=constants.path_to_imgs_dir,
        output_dir=os.path.join(constants.path_to_output_dir, "patches"),
    ):
        """
        create Callable PatchCreator

        Args:
            coco (pycocotools.coco.COCO): loaded coco istance annotation
            img_dir (str): base dir of raw images
            output_dir (str): dst dir for output
        """
        self.img_dir = img_dir
        self.output_dir = output_dir
        self.coco = coco
        self.average_sizes = {}
        self.cat_indices = {}
        for cat in coco.cats.values():
            os.makedirs(
                os.path.join(self.output_dir, f"{cat['id']}-{cat['name']}"),
                exist_ok=True,
            )
            self.cat_indices[cat["name"]] = 0  # count number of objects for indexing
        self.cat_ids = coco.getCatIds()

    def __call__(self, coco_image, dilation=None, blurr=None) -> None:
        """
        create patches from coco image

        Args:
            coco_image: coco.imgs entry
            dilation: amount of morphological mask dilation
            blurr: sigma for a kernel for a Gaussian Blurr
        """
        img_path = os.path.join(self.img_dir, coco_image["file_name"])
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

        anns_ids = self.coco.getAnnIds(imgIds=coco_image["id"], iscrowd=None)
        anns = self.coco.loadAnns(anns_ids)

        for an in tqdm(anns):
            # crop bb from masks and images
            xmin, ymin, w, h = tuple([int(i) for i in an["bbox"]])
            # ensure bb bounds are not outside the image frame
            x1 = max(xmin - self.frame_tol, 0)
            x2 = min(xmin + w + self.frame_tol + 1, img.shape[1] - 1)
            y1 = max(ymin - self.frame_tol, 0)
            y2 = min(ymin + h + self.frame_tol + 1, img.shape[0] - 1)
            mask = self.coco.annToMask(an)
            mask = mask[y1:y2, x1:x2]  # crop to bb + tolerance
            if dilation:
                kernel = np.ones((dilation, dilation))
                mask = cv2.dilate(mask, kernel, np.ones((dilation, dilation)))
            if blurr:
                mask = cv2.GaussianBlur(
                    mask, (blurr * 2 + 1, blurr * 2 + 1), blurr, blurr
                )  # smooth mask for alpha blending
            obj = img[y1:y2, x1:x2].copy()

            # set unmasked corners to transparent
            obj[:, :, 3] = np.where(mask, obj[:, :, 3], 0)

            # save cropped image
            cat = self.coco.loadCats(ids=[an["category_id"]])[0]["name"]
            i = self.cat_indices[cat] + 1
            self.cat_indices[cat] = i
            cv2.imwrite(
                os.path.join(
                    self.output_dir,
                    f"{an['category_id']}-{cat}",
                    f"{cat}(id_{an['category_id']})-{i}-{an['id']}.png",
                ),
                obj,
            )


class CopyPasteGenerator(Dataset):
    """
    Given a directory containing segmented objects (Patches) this class generates
     randomly composed images.

    The class is thought to be able to generate images on-the-fly in order to
     work as dataset class.
    """

    # Perform no data augmentation as default
    AUGMENT = A.Compose(
        [A.NoOp()],
        p=0.0,
    )

    def __init__(
        self,
        patch_pool,
        background_pool,
        scale_augment_dict=None,
        max_n_objs=150,
        augment=None,
        skip_if_overlap_range=(0, 0),
        mask_alpha_blending=0,
    ):
        """
        Initialize abstract CopyPasteGenerator.

        This loads all available background images into RAM and initializes the
        placement frames.
        The objects are loaded, rescaled, and, if specified, augmentations are applied
        to the individual objs and stored in RAM.

        Args:
            patch_pool (dict(PatchPool)): Dictionary of PatchPool objects
            background_pool (BackgroundPool): Backgrounds
            max_n_objs: maximum number of objects placed in one generate image
            n_augmentations: number of augmented version per object
            scale_augment_dict: dict containing each category as key and a dictionary
             definining the number of augmentations (value) at each scale level (key).
             See add_scale_versions for details
            augment: Albumentations data augmentation callable
            skip_if_overlap_range: range of relative object occlusion in which the
             propabilty of a placemente is lineary scaled.
        """
        self.max_n_objs = max_n_objs
        self.augment = self.AUGMENT if not augment else augment
        self.org_pool = patch_pool
        self.mask_alpha_blending = mask_alpha_blending
        if scale_augment_dict:
            self.patches = {cat: {} for cat in self.org_pool.keys()}
            self.add_scaled_versions(scale_augment_dict)
        else:
            self.patches = self.init_default_patch_pool()
        self.backgrounds = background_pool
        self.skip_if_overlap_func = OccludedAreaSkipFunc(*skip_if_overlap_range)

    def init_default_patch_pool(self) -> dict:
        return {
            cat_label: {
                pool.scale: PatchPool.create_from_existing_pool(
                    parent=pool, aug_transforms=self.augment, n_augmentations=1, scale=1
                )
            }
            for cat_label, pool in self.org_pool.items()
        }

    def add_scaled_versions(self, scale_augment_dict) -> None:
        """
        wraps add_scaled_version to acept a dictionary of inputs.

        Args:
            scale_augment_dict: dict containing each category as key and a dictionary
             definining the number of augmentations (value) at each scale level (key).
             See the Example Inputs

        Example Input (1):
            {
                'cat_label_1': {
                    1: 1,  # 1 augmentation per image on scale 1
                    1/2: 5,  # 5 augmentations per image on scale 1/2 (half size)
                    1/4: 10,  # 10 augmentations per image on scale 1/4
                },
                'cat_label_2': {
                    1: 10,  # 10 augmentations per image on scale 1
                    2: 5,  # 5 augmentations per image on scale 2 (double size)
                },
                'cat_label_3': {},  # no augmentations added
            }

        Example Input (2):  # showcases the example for fixed object sizes
            {
                'cat_label_1': {  # will be ignored !
                    500: 1,  # 1 augmentation per image at size 500
                },
                'all_categories': {
                    500: 1,  # 1 augmentation per image at size 500
                    250: 5,  # 5 augmentations per image at size 250
                    125: 10,  # 10 augmentations per image at size 125
                },
        """
        if not "all_categories" in scale_augment_dict.keys():
            for cat_label, d in scale_augment_dict.items():
                if cat_label in self.org_pool.keys():
                    for scale, n_aug in d.items():
                        self.add_scaled_version(cat_label, scale, n_aug)
                else:
                    msg = f"category {str(cat_label)} not in found categories({self.patches.keys()}), please check your category label!"
                    warnings.warn(msg)
        else:
            d = scale_augment_dict["all_categories"]
            for cat_label in self.org_pool.keys():
                for scale, n_aug in d.items():
                    self.add_scaled_version(cat_label, scale, n_aug)

    def add_scaled_version(self, cat, scale=2, n_augmentations=1) -> None:
        """
        adds down-scaled version of defined category to the patch pool

        Args:
            cat (str): category name (without the id) of the category to add
            scale (float): base scale of the patches
            n_augmentations (int): number of augmentations per object
        """
        org_pool = self.org_pool[cat]
        self.patches[cat][scale] = PatchPool.create_from_existing_pool(
            org_pool,
            aug_transforms=self.__class__.AUGMENT,
            n_augmentations=n_augmentations,
            scale=scale,
            mask_alpha_blending=self.mask_alpha_blending,
        )

    def generate(
        self, seed=None
    ) -> (np.ndarray, [np.ndarray], [int], [str], np.ndarray):
        """
        generates image of randomly place objects
        """
        if seed:
            np.random.seed(seed)
        max_n_objs = self.max_n_objs
        img, img_mask, rects = self.backgrounds[-1]
        cats = []
        bboxs = []
        instance_mask = []

        for r in np.random.permutation(len(rects)):
            masks, bbox, cat = self.place_in_rect(img, img_mask, rects[r], max_n_objs)
            max_n_objs -= len(masks)
            cats += cat
            instance_mask += masks
            bboxs += bbox
        return img, instance_mask, bboxs, cats, img_mask

    @abstractmethod
    def place_in_rect(
        self, image, image_mask, frame_rect, max_n_objs
    ) -> ([np.ndarray], [int, int, int, int], [int]):
        """
        Atomic obj placement routine, override this function for specific patterns of
        object placement

        It should paste images (e.g. using
        :func:`copy_and_paste_augm.CopyPasteGenerator.paste_object`) to the background
        image and image_mask, and return the generated annotations.

        Args:
            image (np.ndarray): (partially filled) background image on which objs are
             placed
            image_mask (np.ndarray): (partially filled) mask of the image
            frame_rect ([int, int, int, int]): rectangle defining frame in which the
             objects can be placed (as [xmin, ymin, w, h])
            max_n_objs: maximum number of objs to be placed in this particular frame

        Returns:
            (full_size_masks, bounding_boxes, cats):
                full_size_masks (list(np.ndarray)): list of instance masks
                bounding_boxes (list([int, int, int, int])): bounding boxes of the
                instances
                cats (list(int)): category ids of the instances
        """
        pass

    def __len__(self) -> int:
        """
        dummy length (substitutes epoch length) for Dataset compatibility
        """
        return self.length

    def __getitem__(
        self, seed
    ) -> (np.ndarray, [np.ndarray], [[int, int, int, int]], [int]):
        """
        Generate image with specified seed

        Args:
            seed (int): seed used for np.random.seed

        Returns:
            (img, instance_masks, bboxs, cats):
                img (np.ndarray): generated image (bgr)
                instance_masks (list(np.ndarray)): instance masks
                bboxs (list([int, int, int, int])): bounding boxes of generated
                image
                cats (list(int)): category ids of the instances
        """
        np.random.seed(seed)
        img, instance_masks, bboxs, cats, _ = self.generate()
        return img, instance_masks, bboxs, cats

    def __call__(self, seed=None) -> (np.ndarray, [np.ndarray], [int], [str]):
        """
        Generates an images from the patch and background pool according to the
         logic in place_in_rect.

        Args:
            seed (int): Seed for generating Random number (np.random.seed()). If None
             it is not changed.

        Returns:
            (img, instance_masks, bboxs, cats):
                img (np.ndarray): generated image (bgr)
                instance_masks (list(np.ndarray)): instance masks
                bboxs (list([int, int, int, int])): bounding boxes of generated
                image
                cats (list(int)): category ids of the instances
        """
        img, image_masks, bboxs, cats, _ = self.generate(seed)
        return img, image_masks, bboxs, cats

    def visualize_pool(self) -> None:
        """
        shows examples for each pool
        """
        for cat, d in self.patches.items():
            for scale, pool in d.items():
                pool.visualize_augmentations(3, title=f"{cat} - scale {scale}")

    def get_total_pool_size(self) -> int:
        """
        Returns:
            number of all patches cached
        """
        return sum(
            [
                sum([len(pool) for pool in self.patches[cat].values()])
                for cat in self.patches.keys()
            ]
        )

    @staticmethod
    def paste_object(
        image,
        image_mask,
        obj,
        obj_mask,
        x_min,
        y_min,
        skip_if_overlap_func=None,
    ) -> (np.ndarray, [int, int, int, int]) or None:
        """
        Paste an object on a background image

        Args:
            image (np.ndarray): background image (8 bit bgr)
            image_mask (np.ndarray): background image mask (8 bit greyscale)
             (used for handling overlaps)
            obj (np.ndarray): object (8 bit bgr)
            obj_mask (np.ndarray): mask of the object (8 bit greyscale). The mask is
             used (1) as alpha parameter to wheight alpha blending between the object
             and the background (can be used to smooth image borders) and (2) to
             determine the ground truth mask (mask values of >= 128 are considered
             positive)
            x_min (int): minimum x coord (upper limit) of the object on the background
             image
            y_min (int): minimum y (left limit) coord of the object on the background
             image
            skip_if_overlap_func (func(obj_area, visible_obj_area) -> bool): if the
             func is defined (not None) and evaluates to True the obj is not placed

        Returns:
            inserted object mask (might be partially covered) and coordinates or
             None if the obj was not placed
        """
        # set up coords -> image[y_min : y_max, x_min : x_max] is manipulated
        assert image.dtype == np.uint8 and image.shape[2] == 3
        assert image_mask.dtype == np.uint8
        assert obj.dtype == np.uint8 and obj.shape[2] == 3
        assert obj_mask.dtype == np.uint8

        mask_thresh = 2 ** 7 - 1  # mask threshold

        h, w = obj.shape[0], obj.shape[1]
        x_max = x_min + w
        y_max = y_min + h

        if x_min < 0 or y_min < 0 or x_max >= image.shape[1] or y_max >= image.shape[0]:
            return None

        # handle overlap --> only place the object where there is none yet
        org_area = np.sum(obj_mask >= mask_thresh)
        obj_mask = cv2.bitwise_and(
            obj_mask, cv2.bitwise_not(image_mask[y_min:y_max, x_min:x_max])
        )  # set mask to 0 where other objects have already been placed

        # if skip_if_overlap_func is set and evaluates to True the obj is not pasted
        visible_obj_area = np.sum(obj_mask > 0)
        if skip_if_overlap_func and skip_if_overlap_func(org_area, visible_obj_area):
            return None

        # select background (unchanged area of the image)
        bg = image[y_min:y_max, x_min:x_max]  # relevant rectangle of org image

        # place on obj and mask
        alpha = np.stack([obj_mask / (2 ** 8 - 1)] * 3, axis=2)
        image[y_min:y_max, x_min:x_max] = (
            obj * alpha + image[y_min:y_max, x_min:x_max] * (1 - alpha)
        ).astype(np.uint8)
        obj_mask = cv2.threshold(obj_mask, mask_thresh, 255, cv2.THRESH_BINARY)[1]
        image_mask[y_min:y_max, x_min:x_max] = cv2.bitwise_or(
            image_mask[y_min:y_max, x_min:x_max], obj_mask
        )

        # scale to full image size
        obj_mask_full_size = np.zeros(image.shape[:2], dtype=np.uint8)
        obj_mask_full_size[y_min:y_max, x_min:x_max] = obj_mask

        coords = [x_min, y_min, w, h]

        return obj_mask_full_size, coords


class PatchPool:
    """
    represents the pool of cached object patches for a single object category

    Note, that instantiation loads and keeps ALL images in the specified dir in memory.
    """

    # TODO implement refreshing instances in separate thread
    # TODO implement blurring of the mask for alpha blending

    def  __init__(
        self,
        obj_dir,
        cat_id=-1,
        cat_label=None,
        aug_transforms=None,
        n_augmentations=1,
        min_max_size=50,
        scale=1,
        mask_alpha_blending=0,
        max_size=None,
    ):
        """
        Create PatchPool object from obj_dir to handle RAM caching for the pool of
         patches.

        Args:
            obj_dir (str): parent dir of patches, all  png files are considered
            cat_id (int): category id of the represented patches (label returned
             alongside the generated mask)
            cat_label (str): label of the category (used for selecting pools)
            aug_transforms (Albumentations.Transforms): Albumentations image and mask
             transformation (if None the images is left unchanged)
            n_augmentations (int): number of augmented versions per patch (if set to 0
             only the raw patch pool is created)
            min_max_size (int): min size of the larger patch side
            scale: scale factor by which the patch is resized
            mask_alpha_blending (int): specifies the kernel size for blurring the mask.
             This results in alpha blending between the backround image and the object
             at the object corner.
        """
        self.cat_label = cat_label
        self.cat_id = cat_id
        self.augment = aug_transforms
        self.n_augment = n_augmentations
        self.replace_prob = 0.0
        self.scale = scale
        self.mask_alpha_blending = mask_alpha_blending

        self.files = glob.glob(os.path.join(obj_dir, "*png"))[:max_size]
        self.org_image_pool = [
            self.open_image(file, min_max_size) for file in self.files
        ]
        if n_augmentations < 1:
            self.objs = []
            self.masks = []

            self.image_shapes = np.ndarray([0])
            self.mean_h = 0  # real height (after augmentations)
            self.mean_w = 0  # real width (after augmentations)
            self.mean_max_size = 0  # mean size of largest side
            [self.augment_images(img) for img in self.org_image_pool]
            self.init_image_stats()

    @classmethod
    def create_from_existing_pool(
        cls,
        parent,
        aug_transforms=None,
        n_augmentations=-1,
        scale=0.5,
        mask_alpha_blending=0,
    ):
        """
        create pool from existing raw pool (e.g. for images on another - lower -
        scale). For params see default constructor.
        """
        self = cls.__new__(cls)  # does not call __init__

        self.cat_label = parent.cat_label
        self.cat_id = parent.cat_id
        self.augment = aug_transforms
        self.n_augment = n_augmentations
        if n_augmentations < 1:
            self.augment = None
            self.n_augment = 1
        self.replace_prob = parent.replace_prob
        self.scale = scale
        self.mask_alpha_blending = mask_alpha_blending

        self.objs = []
        self.masks = []

        self.mean_h = 0  # real height (after augmentations)
        self.mean_w = 0  # real width (after augmentations)
        self.mean_max_size = 0  # mean size of largest side

        self.org_image_pool = parent.org_image_pool  # use same original pool
        self.files = parent.files.copy()  # currently unused, required for reloading
        [self.augment_images(img) for img in self.org_image_pool]
        self.init_image_stats()
        return self

    def augment_images(self, obj_raw) -> None:
        """performs self.n_augment random augmentations, appends the resulting
        images to the pool, and crops images and masks to the object bounding
        box"""
        obj_raw, mask_raw = self.split_image_and_mask(obj_raw)

        if self.augment:
            h, w = obj_raw.shape[:2]
            l = max(h, w)
            ymin, xmin = (l - h // 2, l - w // 2)
            ymax, xmax = (ymin + h, xmin + w)

            # pad images
            obj_pad = np.zeros((l * 2, l * 2, 3), dtype=np.uint8)
            obj_pad[ymin:ymax, xmin:xmax, :] = obj_raw
            mask_pad = np.zeros((l * 2, l * 2), dtype=np.uint8)
            mask_pad[ymin:ymax, xmin:xmax] = mask_raw

            # augment
            for _ in range(self.n_augment):
                t = self.augment(image=obj_pad, mask=mask_pad)
                obj = t["image"]
                mask = t["mask"]
                assert obj.dtype == np.uint8
                assert mask.dtype == np.uint8
                self.compress_and_append(obj, mask)
        else:
            self.compress_and_append(obj_raw, mask_raw)

    def compress_and_append(self, img, mask) -> None:
        """
        rescale image and mask and crop to the bounding box of the mask
        """
        if self.scale != 1:  # rescale obj and mask
            img = cv2.resize(
                img,
                (int(img.shape[1] * self.scale), int(img.shape[0] * self.scale)),
                interpolation=cv2.INTER_AREA,
            )
            mask = cv2.resize(
                mask,
                (int(mask.shape[1] * self.scale), int(mask.shape[0] * self.scale)),
                interpolation=cv2.INTER_AREA,
            )
            # mask = cv2.resize(
            #     mask,
            #     (int(mask.shape[1] * self.scale), int(mask.shape[0] * self.scale)),
            #     interpolation=cv2.INTER_AREA,
            # )
        if self.mask_alpha_blending > 1:
            mask = cv2.blur(
                mask,
                (
                    np.random.randint(1, self.mask_alpha_blending),
                    np.random.randint(1, self.mask_alpha_blending),
                ),
            )
            # TODO for some reason the blurring sometimes changes value 0 to 1, dirty fix:
            if np.min(mask) > 0:
                mask = np.where(mask == np.min(mask), 0, mask)

        # plt.imshow(img)
        # plt.show()
        # plt.imshow(mask)
        # plt.show()

        # crop to bounding box of augmented mask
        img, mask = self.crop_to_bb(img, mask)

        # plt.imshow(img)
        # plt.show()
        # plt.imshow(mask)
        # plt.show()

        self.objs.append(img)
        self.masks.append(mask)

    def init_image_stats(self) -> None:
        self.image_shapes = np.array([x.shape[:2] for x in self.masks])
        self.mean_h, self.mean_w = tuple(np.mean(self.image_shapes, axis=0))
        max_sizes = np.max(self.image_shapes)
        self.mean_max_size = np.mean(max_sizes)

    @staticmethod
    def open_image(file, min_max_size) -> np.ndarray:
        """
        open images, handles encoding and scales the larger size up to
         min_max_size
        """
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)

        l = max(img.shape[:2])
        if l < min_max_size:
            rescale_factor = int(2 ** np.ceil(-np.log2(l / min_max_size)))
            img = cv2.resize(
                img,
                (img.shape[1] * rescale_factor, img.shape[0] * rescale_factor),
                interpolation=cv2.INTER_AREA,
            )
        return img

    @staticmethod
    def crop_to_bb(obj, mask) -> (np.ndarray, np.ndarray):
        """
        crop the image and its mask to obj
        """
        x = np.nonzero(np.max(mask, axis=0))
        xmin, xmax = (np.min(x), np.max(x) + 1)
        y = np.nonzero(np.max(mask, axis=1))
        ymin, ymax = (np.min(y), np.max(y) + 1)
        obj = obj[ymin:ymax, xmin:xmax, :]
        mask = mask[ymin:ymax, xmin:xmax]
        return obj, mask

    @staticmethod
    def split_image_and_mask(obj) -> (np.ndarray, np.ndarray):
        """
        split 4 ch (bgra) .png into image (brg) and mask (a)
        """
        rgb = obj[:, :, :3]
        a = obj[:, :, 3]
        return rgb, a

    def replace_image(self, idx) -> None:
        """
        replaces images at idx with new version (in new thread)
        """
        raise NotImplementedError

    def __getitem__(self, idx) -> (np.ndarray, np.ndarray):
        """
        if idx out of bounds a random image is returned. img is returned as bgr.
        """
        if not 0 <= idx < self.__len__():
            idx = np.random.randint(0, self.__len__())
        if np.random.rand() < self.replace_prob:
            self.replace_image(idx)
        return self.objs[idx], self.masks[idx], self.cat_id

    def __len__(self) -> int:
        return len(self.objs)

    def get_mean_height(self) -> float:
        return self.mean_h

    def get_mean_width(self) -> float:
        return self.mean_w

    def visualize_augmentations(self, n_examples=3, title=None):
        """
        show example of augmentations from current image pool
        """
        n_cols = self.n_augment if self.n_augment > 0 else 1
        fig, axs = plt.subplots(
            n_examples, n_cols, figsize=(3 * self.n_augment + 1, 3 * n_examples)
        )
        axs = axs.reshape(-1, self.n_augment)  # in case self.n_augment == 1
        for i in range(n_examples):
            n = np.random.randint(0, self.__len__() / self.n_augment)  # choose obj
            for j in range(self.n_augment):
                k = n * self.n_augment + j
                obj, mask, _ = self[k]
                mask = cv2.bitwise_not(mask)
                obj[mask != 0] = 255
                obj = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)
                obj = self.paste_to_white_square(obj, scale_bar=True)
                axs[i, j].axis("off")
                axs[i, j].imshow(obj)
        if not title:
            title = f"{self.cat_label[k]}({self.cat_id})"  # use saved name as label
        fig.suptitle(title, fontsize=16)
        plt.show()

    def paste_to_white_square(self, img, scale_bar=False):
        """
        paste images into squared frame of fixed size and a scale bar (if specified)
        """
        size = max(int(self.mean_max_size * 2.5), 120)
        sq = np.full((size, size, 3), 255, dtype=np.uint8)
        h, w = img.shape[:2]
        ymin = size // 2 - h // 2
        xmin = size // 2 - w // 2
        sq[ymin : ymin + h, xmin : xmin + w, :] = img
        if scale_bar:
            b_w = 100  # fixed value
            b_h = int(np.ceil(0.04 * size))  # relative to frame size
            b_x = b_y = int(size * 0.9)
            sq[b_y - b_h : b_y, b_x - b_w : b_x, :] = 0
        return sq


class BackgroundPool:
    """
    represents the pool of cached background images.

    Note, that instantiation loads and keeps ALL images in the specified dir into
     memory.
    """

    def __init__(
        self,
        background_dir,
        background_anno="background_anno.json",
        max_resolution=(1800, 1500),
    ):
        """
        loads backgrounds and frame annotation and rescales them to the defined maximum
         resolution

        Args:
            background_dir: path to directory containing usable background images
            background_anno: path to background image frame annotation JSON
            max_resolution (int, int): maximum target resolution of the images (if None
             the original background image resolution is used)
        """
        self.res = max_resolution
        self.background_dir = background_dir
        rects = json.load(open(background_anno))
        self.grid_rects = {
            k: v
            for k, v in rects.items()
            if os.path.exists(os.path.join(self.background_dir, k))
        }

        self.backgrounds = {}
        for k, v in self.grid_rects.items():
            img = cv2.imread(os.path.join(self.background_dir, k))
            assert img.shape[2] == 3
            if self.res:
                # rescale image to max res
                fmax = max(self.res) / max(img.shape[:2])
                fmin = min(self.res) / min(img.shape[:2])
                rescale_factor = min(fmin, fmax)
                img = cv2.resize(
                    img,
                    (
                        int(img.shape[1] * rescale_factor),
                        int(img.shape[0] * rescale_factor),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
                assert max(*self.res) + 1 >= max(img.shape[:2])
                assert min(*self.res) + 1 >= min(img.shape[:2])
            self.backgrounds[k] = img
            # rescale grid rectangle
            for i, l in enumerate(self.grid_rects[k]):
                x, y, w, h = l
                xmax = (x + w) * rescale_factor
                ymax = (y + h) * rescale_factor
                x *= rescale_factor
                y *= rescale_factor
                w = xmax - x
                h = ymax - y
                self.grid_rects[k][i] = [round(x), round(y), round(w), round(h)]

    def __len__(self):
        return len(self.backgrounds)

    def __getitem__(self, idx=-1) -> (np.ndarray, np.ndarray, list):
        """
        Selects the background at position of index (random if index out of bounds) and
         returns it alongside the empty mask and the frame rectangles.
        """
        if not 0 <= idx < self.__len__():
            idx = np.random.randint(0, self.__len__())
        key = list(self.backgrounds.keys())[idx]
        image = self.backgrounds[key].copy()
        rects = self.grid_rects[key]
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        return image, mask, rects


class OccludedAreaSkipFunc:
    """
    Callable modelling a linearly increasing probabilty of object skipping with a
     higher amount of object overlap.
    """

    def __init__(self, min_occ_area, max_occ_area):
        """
        If the relative occluded area is lower than min_are, the p(skip) = 0.
        In [min_occ_area, max_occ_area] the probability increases linearly.
        Above max_occ_area p(skip) = 1.
        Args:
            min_area: Min ocluded area relative to the overall area for an
             overlap to be considered.
            max_area: Max ocluded area relative to the overall area for an
             overlap to be tolerated with a p > 0
        Returns:
            True if the patch is suppossed to be skipped
        """
        self.min = min_occ_area
        self.max = max_occ_area

    def __call__(self, obj_a, vis_a) -> bool:
        if self.max <= self.min:
            return False
        rel_occ = (obj_a - vis_a) / obj_a
        p = rel_occ - self.min
        p /= self.max - self.min
        return np.random.rand() < p


class RandomGenerator(CopyPasteGenerator):
    """
    Places the images randomly with the only restrictions in the case the
     skip_if_overlap function is specified.
    """

    # standard transformation for individual objs completely at random
    AUGMENT = A.Compose(
        [
            A.Rotate(
                limit=360, border_mode=cv2.BORDER_CONSTANT, value=255, mask_value=0, p=1
            ),
            A.transforms.ColorJitter(
                brightness=0.4,
                contrast=0.1,
                saturation=0.2,
                hue=0.06,
                always_apply=False,
                p=0.8,
            ),
        ],
        p=0.85,
    )

    def __init__(
        self,
        patch_pool,
        background_pool,
        scale_augment_dict=None,
        max_n_objs=150,
        skip_if_overlap_range=(0.2, 0.4),
        assumed_obj_size=300 * 300,
        augment=AUGMENT,
        mask_alpha_blending=0,
    ):
        """
        Extends overwritten init methods with the following args

        Args:
            assumed_obj_size: heuristic parameter to specify object density
        """
        super().__init__(
            patch_pool,
            background_pool,
            scale_augment_dict=scale_augment_dict,
            max_n_objs=max_n_objs,
            skip_if_overlap_range=skip_if_overlap_range,
            mask_alpha_blending=mask_alpha_blending,
        )
        self.assumed_obj_size = assumed_obj_size

    def place_in_rect(
        self, image, image_mask, frame_rect, max_n_objs
    ) -> ([np.ndarray], [int, int, int, int], [int]):
        """
        Implementation of abstract place_in_rect function in parent.
        """
        full_size_masks = []
        bounding_boxes = []
        cats = []

        rect_x, rect_y, rect_w, rect_h = frame_rect
        n_obj = rect_h * rect_w // self.assumed_obj_size  # heuristic param for obj size
        n_obj = max(np.random.normal(1, 1) * n_obj, 0)

        for i in range(int(min(max_n_objs, n_obj))):
            cat = np.random.choice([cat for cat in self.patches if self.patches[cat]])
            scale = np.random.choice(list(self.patches[cat]))

            obj, obj_mask, cat = self.patches[cat][scale][-1]
            if rect_w - obj.shape[1] <= 0 or rect_h - obj.shape[0] <= 0:
                continue
            x = np.random.randint(rect_x, rect_x + rect_w - obj.shape[1])
            y = np.random.randint(rect_y, rect_y + rect_h - obj.shape[0])

            pasted = self.paste_object(
                image, image_mask, obj, obj_mask, x, y, self.skip_if_overlap_func
            )
            if pasted:
                full_size_mask, bb = pasted
                full_size_masks.append(full_size_mask)
                bounding_boxes.append(bb)
                cats.append(cat)
        return full_size_masks, bounding_boxes, cats


class CollectionBoxGenerator(CopyPasteGenerator):
    """
    Chooses a single category at a single scale and places objs from this pool in a
     grid layout within the frame.
    """

    # standard transformation for individual objs for box simulation
    AUGMENT = A.Compose(
        [
            A.Rotate(
                limit=15, border_mode=cv2.BORDER_CONSTANT, value=255, mask_value=0, p=1
            ),
            A.transforms.ColorJitter(
                brightness=0.4,
                contrast=0.1,
                saturation=0.2,
                hue=0.05,
                always_apply=False,
                p=0.8,
            ),
            A.transforms.FancyPCA(alpha=0.04, always_apply=False, p=0.25),
        ],
        p=0.85,
    )

    def __init__(
        self,
        patch_pool,
        background_pool,
        scale_augment_dict=None,
        max_n_objs=150,
        skip_if_overlap_range=(0.1, 0.4),
        grid_pos_jitter=(0.1, 0.35),
        space_jitter=(0.6, 1.2),
        augment=AUGMENT,
        mask_alpha_blending=0,
    ):
        """
        Extends overwritten init methods with the following args

        Args:
            grid_pos_jitter ((float, float)): heuristic parameter to specify jitter when
             placing objects on the grid. The actual parameter is sampled from the given
             interval for each generated image. Hence, the average amount of jitter
             varies between generate images.
            space_jitter ((float, float)): heuristic parameter to specify variation in
             the assumed object size and, hence, the density of object placement.
             Note, that this might interfere with the skip_if_overlap_func.
        """
        super().__init__(
            patch_pool,
            background_pool,
            scale_augment_dict=scale_augment_dict,
            max_n_objs=max_n_objs,
            skip_if_overlap_range=skip_if_overlap_range,
            augment=augment,
            mask_alpha_blending=mask_alpha_blending,
        )
        assert grid_pos_jitter[1] >= grid_pos_jitter[0]
        self.grid_jitter = grid_pos_jitter
        assert space_jitter[1] >= space_jitter[0]
        self.space_jitter = space_jitter

    def place_in_rect(
        self, image, image_mask, frame_rect, max_n_objs
    ) -> ([np.ndarray], [int, int, int, int], [int]):
        """
        Implementation of abstract place_in_rect function in parent.
        """
        # choose random cat and scale (equival p for each cat and each scale within each cat)
        cat = np.random.choice([cat for cat in self.patches if self.patches[cat]])
        scale = np.random.choice(list(self.patches[cat]))
        pool = self.patches[cat][scale]
        grid_jitter = np.random.uniform(*self.grid_jitter)

        rect_x, rect_y, rect_w, rect_h = frame_rect
        av_w = pool.get_mean_width() * np.random.uniform(*self.space_jitter)
        av_h = pool.get_mean_height() * np.random.uniform(*self.space_jitter)
        grid_n_x = rect_w // int(av_w)
        grid_n_y = rect_h // int(av_h)

        full_size_masks = []
        bounding_boxes = []
        cats = []

        for i, j in itertools.product(range(grid_n_y), range(grid_n_x)):
            if len(full_size_masks) >= max_n_objs:
                break
            obj, obj_mask, cat = pool[-1]  # get random patch

            # calculate location of the center
            x = rect_x + j * av_w + av_w // 2
            y = rect_y + i * av_h + av_h // 2

            # calculate left / lower limit and add some jitter
            x -= np.random.normal(obj.shape[1] / 2, grid_jitter)
            y -= np.random.normal(obj.shape[0] / 2, grid_jitter)
            x, y = int(x), int(y)

            # check if the instance is still in the frame
            if (
                x + obj.shape[1] > rect_x + rect_w
                or y + obj.shape[0] > rect_y + rect_h
                or x < rect_x
                or y < rect_y
            ):
                continue

            pasted = self.paste_object(
                image, image_mask, obj, obj_mask, x, y, self.skip_if_overlap_func
            )
            if pasted:
                full_size_mask, bb = pasted
                full_size_masks.append(full_size_mask)
                bounding_boxes.append(bb)
                cats.append(cat)
        return full_size_masks, bounding_boxes, cats


def main():
    background_pool = BackgroundPool(
        background_dir=os.path.join(constants.path_to_copy_and_paste, "backgrounds"),
        background_anno=os.path.join(
            constants.path_to_copy_and_paste, "backgrounds/background_anno.json"
        ),
        max_resolution=(1800, 1500),
    )

    obj_dir = os.path.join(constants.path_to_copy_and_paste, "beetles")
    dirs = [os.path.basename(x) for x in glob.glob(os.path.join(obj_dir, "*"))]
    cat_ids = {  # define ids returned for each patch class
        "1035185": 0,
        "1035542": 0,
        "1036154": 2,
        "4470539": 1,
        "foo": 42,
    }

    cat_labels = dirs
    # create base patch pool dict
    patch_pool = {
        cat_label: PatchPool(
            os.path.join(obj_dir, cat_label),
            cat_id=cat_ids[cat_label],
            cat_label=cat_label,
            aug_transforms=None,
            n_augmentations=0,  # only create base Pool
            scale=1,
        )
        for cat_label in cat_labels
    }

    scales = {  # define scales of each category
        "1035185": {0.4: 1},
        "1035542": {0.4: 2},
        "1036154": {0.3: 1},
        "4470539": {0.5: 2},
        "foo": {},
    }

    cpg = CollectionBoxGenerator(
        patch_pool, background_pool, scales, max_n_objs=150, mask_alpha_blending=3
    )
    # cpg.visualize_pool()
    for i in range(10):
        img, image_masks, bboxs, cats, image_mask = cpg.generate(i)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(i)
        plt.show()
        cv2.imwrite(f"C:/users/sebas/Desktop/output/image_{i}.png", img)
        cv2.imwrite(f"C:/users/sebas/Desktop/output/mask_{i}.png", image_mask)


if __name__ == "__main__":
    main()
