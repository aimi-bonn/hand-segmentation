"""
functionality used in data preprocessing
"""
import os
import numpy as np
import cv2
import pandas as pd
import scipy
from glob import glob
import skimage
from skimage.filters import *
from tqdm import tqdm
from google.colab.patches import cv2_imshow


def segment_bugs_from_crops(img, gaus_sigma=1 / 40, average_blur_size=1 / 20):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # legs
    legs = gray.copy()
    legs_background = scipy.ndimage.gaussian_filter(
        legs, sigma=round(img.shape[1] * gaus_sigma)
    )
    legs = cv2.subtract(legs_background, legs)
    legs = (legs > threshold_otsu(legs)).astype(np.uint8)

    # body
    body = cv2.blur(gray, (round(img.shape[1] * average_blur_size),) * 2)
    body = body < threshold_minimum(body)
    body = scipy.ndimage.binary_fill_holes(body).astype(np.uint8)

    # whole bug
    close_size = max(7, img.shape[1] // 80)
    mask = cv2.bitwise_or(body, legs)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    mask = mask.astype(np.uint8)

    cont, hir = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(cnt) for cnt in cont])
    idx = areas.argmax()

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(
        image=mask,
        contours=cont,
        contourIdx=idx,
        color=(255),
        thickness=-1,
        hierarchy=hir,
        maxLevel=1,
    )
    return mask


def demonstrate_mod_IOU(path, predictor):
    img = cv2.imread(path)
    # img = cv2.resize(img, (round(img.shape[1]*0.5),
    #                                     round(img.shape[0]*0.5)
    #         ))
    cv2_imshow(img)

    outputs = predictor(img)
    pred = outputs["instances"].pred_masks.cpu().numpy()[0, :, :]
    mask = segment_bugs_from_crops(img)

    comp = img.copy()
    comp[:, :, 1] += (pred * 128).astype(np.uint8)
    comp[:, :, 2] += (mask // 2).astype(np.uint8)
    cv2_imshow(comp)

    # visualize IOU
    img1 = pred.copy()
    img2 = mask.copy()
    img1 = img1.astype(np.bool)
    img2 = img2.astype(np.bool)

    intersection = img1 * img2
    union = img1 + img2

    cv2_imshow(intersection.astype(np.uint8) - union.astype(np.uint8))

    # visualize mod IOU ignoring pixel at the edges
    kernel = np.ones((9, 9), np.uint8)

    img1 = pred.copy()
    img2 = mask.copy()

    img1 = img1.astype(np.uint8) // img1.max()
    img2 = img2.astype(np.uint8) // img2.max()

    intersection1 = img1 * cv2.dilate(img2, kernel, iterations=1).astype(np.bool)
    intersection2 = img2 * cv2.dilate(img1, kernel, iterations=1).astype(np.bool)
    intersection = (intersection1 + intersection2) > 0
    union = (img1 + img2) > 0

    cv2_imshow(intersection.astype(np.uint8) - union.astype(np.uint8))


def i_o_u(img1, img2):
    """
    Intersection over Union for binary mask
    """
    img1 = img1.astype(np.bool)
    img2 = img2.astype(np.bool)

    intersection = img1 * img2
    union = img1 + img2

    return np.sum(intersection) / np.sum(union)


def mod_i_o_u(img1, img2):
    """
    Intersection over Union modified to ignore some edge pixels
    """
    kernel = np.ones((9, 9), np.uint8)

    img1 = img1.astype(np.uint8) // img1.max()
    img2 = img2.astype(np.uint8) // img2.max()

    intersection1 = img1 * cv2.dilate(img2, kernel, iterations=1).astype(np.bool)
    intersection2 = img2 * cv2.dilate(img1, kernel, iterations=1).astype(np.bool)
    intersection = (intersection1 + intersection2) > 0
    union = (img1 + img2).astype(np.bool)

    return np.sum(intersection) / np.sum(union)
