import streamlit as st
from PIL import Image
import cv2
import random
import numpy as np
import pandas as pd
from pathlib import Path
import glob
from src import get_mask, get_labels

@st.cache
def filter_diamonds(filenames, options):
    filter_filenames = []
    if len(options) > 0:
        for filename in filenames:
            ex = True
            labels = get_labels(Path(img_dir + '/' + filename), classes)
            for val in options:
                if val not in labels:
                    ex = False
            if ex: filter_filenames.append(filename)
    else:
        filter_filenames = filenames
        
    return filter_filenames

img_dir = "./2021-08-10-Examples"

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
st.title('Diamond Example')

# filter
options = st.multiselect(
    'Filter by',
    classes.values()
)

filenames = [file.split('/')[-2] for file in glob.glob("./2021-08-10-Examples/*/Darkfield_EF.jpg")]

filter_filenames = filter_diamonds(filenames, options)

if len(filter_filenames) == 0:
    not_found = "vis_utils/images/hqdefault.jpg"
    st.image(not_found)
else:    
    # select diamond
    diamond = st.selectbox(
        'Choose a Diamond',
        filter_filenames)

    # take img and masks
    img     = np.array(Image.open(img_dir + '/' + diamond + '/' + "Darkfield_EF.jpg"))
    mask    = get_mask(Path(img_dir + '/' + diamond), classes_id)
    labels  = get_labels(Path(img_dir + '/' + diamond), classes)
        
    # labels
    st.text('In this diamond there are: ' + ', '.join(str(l) for l in labels))
    
    # radio
    radio = st.radio(
    "Choose img or svg",
    ('img', 'svg'))

    # sidebar
    st.sidebar.title("Masks")

    # svg
    svg  = np.array(Image.open(img_dir + '/' + diamond + '/' + "file.png"))

    # plot masks on img
    for i, cl in enumerate(classes.values()):

        ag = st.sidebar.checkbox(cl, value= cl in known_classes)
        if ag == True:
            tmp_mask = np.zeros(img.shape, dtype=np.uint8)
            tmp_mask[mask[i] > 0] = palette[i]
            img[mask[i] > 0] = img[mask[i] > 0]//2 + tmp_mask[mask[i] > 0]//2

    # display img or svg
    if radio == 'img':
        st.image(img)
    else:
        st.image(svg)
    
