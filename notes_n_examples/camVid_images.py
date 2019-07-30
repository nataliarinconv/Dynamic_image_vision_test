from os import listdir;
from PIL import Image as PImage;
import natalialibNew as nlib;
import cv2;
import numpy as np;

def loadImages(path):
    # return array of images

    imagesList = sorted(listdir(path));
    loadedImages = [];

    for image in imagesList:
        img = cv2.imread(path + image);
        loadedImages.append(img);

    return loadedImages;

path = "./not_labelled/";
l_path = "./labelled/";

#images in an array
not_labelled = loadImages(path);
labelled = loadImages(l_path);
 
# labelled.astype(np.uint8)

nlib.show_video(not_labelled,100);
nlib.show_video(labelled,100);

np.savez('Camvid_not_labelled', not_labelled);
np.savez('Camvid_labelled', labelled);
# so in Camvid_full is both the not labelled and the labelled full video 