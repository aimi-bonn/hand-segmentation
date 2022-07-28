import os
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import cv2
from lib import cv2_topology_handler
from lib import constants

DESCRIPTION = "Data set for segmenting insects"
DEFAULT_LICENSE = {
    "id": 1,
    "name": "Attribution-NonCommercial-ShareAlike License",
    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
}

__all__ = ["CocoDataset"]


class CocoDataset:
    """
    builds coco dataset iteratively to obtain coco annotation file
    """

    def __init__(self, path_to_existing_coco=None, info=None, license=None):
        """
        creates coco_dataset instance based on existing coco annotation or a dummy file
         (if path_to_existing_coco = None)
        """
        self.info = {}
        self.licenses = []
        self.images = []
        self.categories = []
        self.annotations = []
        self.n_images = 0
        self.n_categories = 0
        self.n_annotations = 0
        if not info:
            self.info = self.create_coco_info(
                contrib=["Sebastian Rassmann"], version="v0.0.0", descr=DESCRIPTION
            )
        if not license:
            self.licenses = [DEFAULT_LICENSE]
        if path_to_existing_coco:
            ann = COCO(path_to_existing_coco)
            self.images = list(ann.imgs.values())
            self.categories = list(ann.cats.values())
            self.annotations = list(ann.anns.values())
        self.category_name_id_dict = {
            entry["name"]: entry["id"] for entry in self.categories
        }
        self.tmp_output_dir = os.path.join(
            constants.path_to_copy_and_paste, "tmp", "tmp_coco_anno.json"
        )

    def find_cat_name_of_id(self, cat_id) -> list:
        return list(self.category_name_id_dict.keys())[
            list(self.category_name_id_dict.values()).index(cat_id)
        ]

    def add_annotation_from_binary_mask(
        self,
        mask,
        image_name,
        category_name,
        super_category_name="",
        img_license=1,
        min_area=0,
    ) -> None:
        """
        creates COCO formatted instance annotation and add it to the coco file.

        Objects are separated if they are in different mask images or if they are not
        connected within the same mask.

        Args:
            mask (np.ndarray  or str): mask image or path to the mask
            image_name (str): name of the image file
            category_name (str): name of category
            super_category_name (str): name of super category
            img_license (int): img license id
            min_area (int): minimum object area to be considered
        """
        if isinstance(mask, str):
            mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        image_id = len(self.images)
        self.images.append(
            self.create_coco_image(
                img_license, image_name, mask.shape[0], mask.shape[1], image_id
            )
        )
        if category_name not in self.category_name_id_dict.keys():
            cat_id = len(self.category_name_id_dict) + 1
            self.categories.append(
                self.create_coco_category(category_name, cat_id, super_category_name)
            )
            self.category_name_id_dict[category_name] = len(self.categories)
        else:
            cat_id = self.category_name_id_dict[category_name]

        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) > 0:
            i = 0
            while True:  # all contours on first hierarchy level
                c = cv2_topology_handler.handle_contour_topology(
                    contours, hierarchy, i
                )  # exclude holes
                area = cv2.contourArea(c)
                if area > min_area:
                    self.annotations.append(
                        CocoDataset.create_coco_segmentation(
                            [self.cv2_contour_to_coco_annotation(c)],
                            [*cv2.boundingRect(c)],
                            area,
                            len(self.annotations),
                            image_id,
                            cat_id,
                            is_crowd=0,
                        )
                    )
                i = hierarchy[0, i, 0]  # get next contour
                if i < 0:
                    break

    def add_annotations_from_instance_mask(
        self,
        mask,
        image_name,
        category_names,
        img_license=1,
        min_area=0,
    ):
        """
        creates COCO formatted instance annotation and add it to the coco file

        If a single mask is provided objects are only separated if intensity values
         (greyscale) differ and the greyscale value is used to index category_name
         (Note, that, in this case, the 1-based index values in the mask are converted
         into 0-based indices for accessing the list).
        If a list of masks is provided each mask is thought to contain a single
        instance and the index of the mask is used to index category_names.

        Args:
            mask (np.ndarray or list(np.ndarray) or str): mask image, list of mask
             images, or str as path to the mask
            image_name (str): name of the image file
            category_names (str): name of categories as list
            img_license (int): img license id
            min_area (int): minimum object area to be considered
        """
        if isinstance(mask, str):
            mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        if isinstance(mask, list):  # project to single value encoded image
            if len(mask):
                # stack images and assign instance id as value
                mask = np.stack(mask).astype(np.bool) * np.arange(
                    1, len(mask) + 1
                ).reshape(-1, 1, 1)
                assert np.all(
                    np.max(mask, axis=0) == np.sum(mask, axis=0)
                ), "provided instance masks overlap"
                mask = np.max(mask, axis=0).astype(np.uint32)
            else:
                mask = np.zeros((1, 1))

        image_id = len(self.images)
        self.images.append(
            self.create_coco_image(
                img_license, image_name, mask.shape[0], mask.shape[1], image_id
            )
        )
        # create non existing categories
        existing_cats = self.category_name_id_dict.keys()
        for n in set(category_names):
            if n not in existing_cats:
                self.categories.append(
                    self.create_coco_category(n, len(self.categories) + 1, "")
                )
                # TODO implement retrieval of supercat
                self.category_name_id_dict[n] = len(self.categories)

        for value in np.unique(mask):
            if value == 0:  # ignore background
                continue
            cont, hir = cv2.findContours(
                (mask == value).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            ann = self.create_coco_segmentation_single_instance(
                cont,
                hir,
                entry_id=len(self.annotations),
                image_id=image_id,
                category_id=self.category_name_id_dict[category_names[value - 1]],
                min_area=min_area,
            )
            if ann:
                self.annotations.append(ann)

    def to_json(self, path=None):
        """
        dump obj to coco-compatible JSON String
        """
        import json

        coco_dict = {
            "info": self.info,
            "categories": self.categories,
            "images": self.images,
            "annotations": self.annotations,
            "licenses": self.licenses,
        }
        if path:
            with open(path, "w+") as f:
                f.write(json.dumps(coco_dict, indent=4, sort_keys=False))

    def show_annotations(self, data_dir=constants.path_to_imgs_dir, cat_names="all"):
        """
        verification method, dumps itself to json and reloads using COCO
        """
        self.to_json(self.tmp_output_dir)
        coco = COCO(self.tmp_output_dir)
        if cat_names == "all":
            cat_ids = coco.getCatIds()
        else:
            cat_ids = coco.getCatIds(catIds=cat_names)
        for i in coco.imgs:
            annIds = coco.getAnnIds(imgIds=i, catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(annIds)
            img = cv2.imread(os.path.join(data_dir, coco.imgs[i]["file_name"]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 15))
            plt.axis("off")
            plt.imshow(img)
            coco.showAnns(anns)
            plt.show()

    def get_categories_as_list(self):
        return [cat_dict["name"] for cat_dict in self.categories]

    def get_categories(self):
        return self.categories.copy()

    @staticmethod
    def create_coco_segmentation_single_instance(
        contours,
        hierarchy,
        entry_id,
        image_id,
        category_id,
        min_area=0,
        is_crowd=0,
    ) -> dict:
        """
        Create coco segmentation from cv2 contours correcting for topology (e.g.
        holes encapsulated in the object)

        Contours on the same hierarchy level are assumed to be the same instance.
        """
        segs = []
        area = 0
        outline = []
        i = 0  # outer objects is at zero in hierarchy
        while True:
            if len(contours[i]) > 2:  # contour is valid
                c = cv2_topology_handler.handle_contour_topology(
                    contours, hierarchy, i
                )  # exclude holes
                segs.append(
                    CocoDataset.cv2_contour_to_coco_annotation(c)
                )  # add contour
                area += cv2.contourArea(c)
                x, y, w, h = cv2.boundingRect(c)
                outline.append([x, y, x + w, y + h])
            i = hierarchy[0, i, 0]
            if i == -1:
                break  # stop searching once there are no more contours on the same level
        # evaluate joint metric
        if area < min_area:
            return None
        outline = np.array(outline)
        x = int(np.min(outline[:, 0]))
        y = int(np.min(outline[:, 1]))
        w = int(np.max(outline[:, 2])) - x
        h = int(np.max(outline[:, 3])) - y
        return CocoDataset.create_coco_segmentation(
            segs, [x, y, w, h], area, entry_id, image_id, category_id, is_crowd
        )

    ## --- Methods to handle COCO compatible entry formatting ---

    @staticmethod
    def cv2_contour_to_coco_annotation(contour):
        return contour.ravel().tolist()

    @staticmethod
    def create_coco_segmentation(
        poly, bbox, area, entry_id, image_id, category_id, is_crowd=0
    ):
        return {
            "id": entry_id,
            "image_id": image_id,
            "category_id": category_id,
            "iscrowd": is_crowd,
            "area": area,
            "bbox": bbox,
            "segmentation": poly,
        }

    @staticmethod
    def create_coco_category(cat_name, cat_id, super_cat=""):
        return {"supercategory": super_cat, "id": cat_id, "name": cat_name}

    @staticmethod
    def create_coco_image(license_id, filename, height, width, image_id):
        return {
            "img_license": license_id,
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id,
        }

    @staticmethod
    def create_coco_info(
        descr="desc",
        url="url",
        version="v0.0.0",
        year=None,
        contrib="cont",
        date_created=None,
    ):
        if not year or not date_created:
            from datetime import date

            year = year if year else date.today().strftime("%Y")
            date_created = (
                date_created
                if date_created
                else date.today().strftime("%Y/%m/%d-%H:%M:%S")
            )
        return {
            "description": descr,
            "url": url,
            "version": version,
            "year": year,
            "contributor": contrib,
            "date_created": date_created,
        }


def main():
    from lib import constants
    import pandas as pd

    c = CocoDataset(os.path.join(constants.path_to_anno_dir, "butterfly_anno.json"))
    d = os.path.join(constants.path_to_data_dir, "bug_labelling.csv")
    d = pd.read_csv(d)
    for _, row in d.iterrows():
        c.add_annotation_from_binary_mask(
            os.path.join(constants.path_to_masks_dir, row["mask"]),
            row["crop_image_name"],
            row["rough_class"],
            row["rough_class"],
            min_area=row["min_area"],
        )
    # c.show_annotations(cat_names=["bug"])
    c.to_json(os.path.join(constants.path_to_anno_dir, "all_anno.json"))


if __name__ == "__main__":
    main()
