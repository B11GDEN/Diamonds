import streamlit as st
import cv2
import random
import numpy as np
import pandas as pd
from pathlib import Path
import glob
from scripts.generate_masks import make_mask

known_classes = ["Internal graining", "Surface graining", "Chip", "Bruise", "Twinning wisp", "Needle", "Pinpoint", "Feather", "Cloud", "Crystal"]

classes = {
    1 : "Crystal",
    2 : "Cloud",
    3 : "Feather",
    4 : "Pinpoint",
    5 : "Needle",
    6 : "Twinning wisp",
    7 : "Bruise",
    8 : "Cavity",
    9 : "Chip",
    10: "Indented natural",
    11: "Knot",
    12: "Laser drill hole",
    13: "Natural",
    14: "Etch channel",
    15: "Polish lines",
    16: "Extra facet",
    17: "Surface graining",
    18: "Bearding",
    19: "Grain center",
    20: "Internal graining",
    21: "Lizard skin",
    22: "Pits",
    23: "Burn",
    24: "Nick"
}

classes_id = list(classes.keys())

# generate random palette for masks
random.seed(42)
palette = [(random.randint(200,255), random.randint(200,255), random.randint(200,255)) for i in range(len(classes_id))]

# title
st.title('Mask Example')

# select diamond
filenames = [file.split('/')[-2] for file in glob.glob("./2021-08-10-Examples/*/Darkfield_EF.jpg")]

diamond = st.selectbox(
    'Choose a Diamond',
    filenames)

# take img and masks
img_dir = "./2021-08-10-Examples"
ori     = cv2.imread(img_dir + '/' + diamond + '/' + "Darkfield_EF.jpg")
mask    = make_mask(Path(img_dir + '/' + diamond), classes_id)

# sidebar
st.sidebar.title("Masks")

ag = st.sidebar.checkbox("img", value=True)
if ag == True:
    img = ori
else:
    img = np.zeros(ori.shape, dtype=np.uint8)
    
ag = st.sidebar.checkbox("svg", value=True)
if ag == True:
    svg  = cv2.imread(img_dir + '/' + diamond + '/' + "file.png")
else:
    svg = np.zeros(ori.shape, dtype=np.uint8)

# plot masks on img and svg
for i, cl in enumerate(classes.values()):
    
    ag = st.sidebar.checkbox(cl, value= cl in known_classes)
    if ag == True:
        tmp_mask = np.zeros(img.shape, dtype=np.uint8)
        tmp_mask[mask[i] > 0] = palette[i]
        img[mask[i] > 0] = img[mask[i] > 0]//2 + tmp_mask[mask[i] > 0]//2
        svg[mask[i] > 0] = svg[mask[i] > 0]//2 + tmp_mask[mask[i] > 0]//2

col1, col2 = st.columns(2) 

# Original image on the left
with col1:
    st.header("Original")
    st.image(img)
# SVG image on the right    
with col2:
    st.header("SVG")
    st.image(svg)
    
