import os.path

import cv2
import numpy as np
from glob import glob

l = glob("/home/rassman/bone2gene/masking/data/val/*_ori.png")

# for path in l:
#     img = cv2.imread(path)
#     mask = path.replace("_ori.png", "_bin_pruned.png")
#
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
#     img[:, :, -1] = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
#
#     cv2.imwrite(path.replace("_ori.png", ".png"), img)

for path in glob("/home/rassman/Desktop/val/*"):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    name = os.path.basename(path)
    img = cv2.imread(
        f"/home/rassman/bone2gene/data/annotated/uts_magdeburg/{name}",
        cv2.IMREAD_COLOR,
    )
    if img is None:
        print(path)
        continue

    cv2.imwrite(
        f"/home/rassman/bone2gene/masking/data/train2/{name.replace('.png', '_ori.png')}",
        img,
    )
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    img[:, :, -1] = mask

    cv2.imwrite(f"/home/rassman/bone2gene/masking/data/train2/{name}", img)

    cv2.imwrite(
        f"/home/rassman/bone2gene/masking/data/train2/{name.replace('.png', '_bin_pruned.png')}",
        mask,
    )
