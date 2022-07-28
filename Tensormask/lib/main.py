from lib.copy_and_paste_augm import CopyPasteGenerator, CollectionBoxGenerator
from lib.coco_handler import CocoDataset
from lib import constants
from lib.copy_and_paste_augm import *
import cv2

import os


def main():
    obj_dir = os.path.join(constants.path_to_copy_and_paste, "objs")
    dirs = [os.path.basename(x) for x in glob.glob(os.path.join(obj_dir, "*"))]
    cat_ids = [s.split("-")[0] for s in dirs]
    cat_labels = [s.split("-")[1] for s in dirs]
    # create patch pool dict
    patch_pool = {
        cat_label: PatchPool(
            os.path.join(obj_dir, f"{cat_id}-{cat_label}"),
            cat_id=cat_id,
            cat_label=cat_label,
            aug_transforms=None,
            n_augmentations=0,  # only create Pool
            scale=1,
            )
        for cat_id, cat_label in zip(cat_ids, cat_labels)
        }

    background_pool = BackgroundPool(
        background_dir=os.path.join(constants.path_to_copy_and_paste, "backgrounds"),
        background_anno="background_anno.json",
        max_resolution=(1800, 1500),
        )

    d = {
        "Mesembryhmus_purpuralis": {1: 5, 0.25: 5},
        "Smerinthus_ocellata": {1: 5, 1 / 2: 5},
        "Acherontia_atroposa": {1: 5, 1 / 4: 3},
        "bug_proxy_2": {1: 3},
        "bug_proxy_3": {1: 3},
        "bug_proxy_1": {1: 3, 4: 2},
        "Trichotichnus": {1: 3, 4: 2},
        }
    cpg_coco_train = CocoDataset(
        info=CocoDataset.create_coco_info(
            descr="""
        Image set created by Copy and Paste data augmentation
    """,
            contrib="Sebastian Rassmann",
            )
        )
    cpg = CollectionBoxGenerator(patch_pool, background_pool, d, max_n_objs=150)
    for i in range(5):
        img, instance_masks, bboxs, cats, image_mask = cpg.generate()
        img_name = f"train-box-{i}.png"
        cv2.imwrite(
            os.path.join(constants.path_to_copy_and_paste, "output", "train", img_name),
            img,
            )
        cpg_coco_train.add_annotations_from_instance_mask(
            instance_masks, img_name, cats
            )
        # cpg_coco_train.show_annotations(
        #     os.path.join(constants.path_to_copy_and_paste, "output", "train")
        # )


if __name__ == "__main__":
    main()
