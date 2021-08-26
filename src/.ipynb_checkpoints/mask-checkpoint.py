""" small script to generate masks """

import os
import cv2
import json
import numpy as np
from PIL import Image

def get_mask(
    img_dir,
    classes_id=[1, 2, 3, 4, 5, 6, 7, 9, 17, 20],
    dilate_kernel = (7, 7)
):
    """Args:
        img_dir (pathlib.Path): path to single folder with diamond files
        classes_id (list): id of classes for which the mask is parsed
        dilate_kernel (int, int): dilate kernel
    
    Returns:
        mask as np.arrays, format CHW"""
    
    formats = [
        'com.octonus.CInclusionRepresentationCloudFormat/1.0',
        'com.octonus.CInclusionRepresentationContoursFormat/1.0',
        'com.octonus.CInclusionRepresentationFeatherFormat/1.0',
        'com.octonus.CInclusionRepresentationNeedleFormat/1.0',
        'com.octonus.CInclusionRepresentationPinpointFormat/1.0',
    ]
    
    def _extract_contour(representation):
        pnts = []

        tmp_mask = np.zeros((height, width), dtype=np.uint8)

        for r in representation:
            if r["serid"] in formats[0:4]:
                pnts.append(tuple(r["contourContainer"]["contours"]["Values"][0]["points"]["points"]))
            elif r["serid"] in formats[4]:
                x, y, rad = int(r["centerX"]), int(r["centerY"]), int(r["radius"])
                tmp_mask = cv2.circle(tmp_mask, (x, y), rad, 255, -1)
            else:
                continue
            
        un_pnts = set(pnts)
        
        for up in un_pnts:
            upd = list(zip(up[::2], up[1::2]))
            path = np.array(upd, dtype="int").reshape(-1, 2)
            tmp_mask = cv2.fillConvexPoly(tmp_mask, path, 255)

        return tmp_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dilate_kernel)
    
    mask_pathes = list(img_dir.glob("*.msainclusions"))
    width, height = Image.open(img_dir / "Darkfield_EF.jpg").size
    
    mask = np.zeros((len(classes_id), height, width), dtype=np.uint8)

    for mask_path in mask_pathes:
        mask_data = json.load(mask_path.open())
        vals = mask_data["Data"]["data"]["inclusions"]["Values"]
        for v in vals:
            reprs = v["representations"]["Values"]
            if v["type"] not in classes_id: continue
            mask_id  = classes_id.index(v["type"])
            tmp_mask = _extract_contour(reprs)
            tmp_mask = cv2.dilate(tmp_mask, kernel)
                
            mask[mask_id] |= tmp_mask
            
            
    return np.flip(mask, axis=1)

def get_labels(img_dir, classes):
    """Args:
        img_dir (pathlib.Path): path to single folder with diamond files
        classes (dict): a dictionary of classes we are interested in
    
    Returns:
        labels: list of diamond incusions"""
    
    labels = set()
    mask_pathes = list(img_dir.glob("*.msainclusions"))
    
    for mask_path in mask_pathes:
        mask_data = json.load(mask_path.open())
        vals = mask_data["Data"]["data"]["inclusions"]["Values"]
        for v in vals:
            reprs = v["representations"]["Values"]
            labels.add(classes[v["type"]])
            
    return labels

