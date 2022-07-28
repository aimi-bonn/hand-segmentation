"""
defines folder structure of the project. Adapt if structure need to be changed.
"""
import os


# use path relative to constants.py as reference
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

path_to_data_dir = os.path.join(project_dir, "data")

path_to_output_dir = os.path.join(project_dir, "output")

path_to_imgs_dir = os.path.join(path_to_data_dir, "raw", "imgs")

path_to_masks_dir = os.path.join(path_to_data_dir, "raw", "masks")

path_to_anno_dir = os.path.join(path_to_data_dir, "raw", "annotations")

path_to_copy_and_paste = os.path.join(path_to_data_dir, "copy_and_paste")
