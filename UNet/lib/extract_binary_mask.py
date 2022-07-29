import numpy as np
import cv2
from tqdm import tqdm
from glob import glob


def main():
    for f in tqdm(glob("/home/rassman/bone2gene/data/masks/unet/*/*.png")):
        if "vis" in f or "bone_age" in f:
            continue

        raw_prediction = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        binary = (raw_prediction > 0).astype(np.uint8)
        conts, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        conts = sorted(conts, key=lambda x: cv2.contourArea(x), reverse=True)
        hand = None
        for c in conts:  # extract the largest area containing high confidence
            pred = cv2.drawContours(np.zeros_like(raw_prediction), [c], -1, 255, -1)
            if np.any(pred * (raw_prediction > 250)):
                hand = pred
                break
        if hand is None:
            c = conts[0]
            hand = cv2.drawContours(np.zeros_like(raw_prediction), [c], -1, 255, -1)
        hand = hand.astype(np.uint8)

        cv2.imwrite(f, hand)


if __name__ == "__main__":
    main()
