# match.py
# Object-to-Scene Matcher
#
# Author: James Zhang (jameszha@andrew.cmu.edu)
# Date: Mar. 1, 2020
#
# Description: 
# Given an object image and a scene image, match attempts to locate the object within the scene.
# The position (x, y) of the matched object is returned.
# Matches can be outputted to an output image for visualization.
# 
# Currently, match supports the ORB (Oriented FAST and Rotated BRIEF) and SIFT (Scale-Invariant Feature Transform) 
# detectors. 
#
# Usage: match.py [-h] -i INPUT -s SCENE [-o OUTPUT] [-d {ORB,SIFT}]
# optional arguments:
#   -h, --help                              show this help message and exit
#   -i INPUT, --input INPUT                 Input object image filename
#   -s SCENE, --scene SCENE                 Input scene image filename
#   -o OUTPUT, --output OUTPUT              Output image (with matches shown) filename
#   -d {ORB,SIFT}, --detector {ORB,SIFT}    Choice of feature detector
#
#
# References:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html
# Lowe, David. Distinctive Image Features from Scale-Invariant Keypoints. University of British Columbia. 2004.
#
# FAIR USE NOTICE: This file contains copyrighted material the use of which has not been specifically authorized by the copyright
# owner. This material is to be used for research and educational purposes only. We consider use of any such copyrighted material 
# to be fair use per section 107 of the US Copyright Law.
#

# Support libraries
import argparse
import sys
import time

# Image processing libraries
# $ pip3 install opencv-contrib-python==3.4.2.16
import numpy as np 
import cv2

# Error print
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# Text color codes for fancy printouts
BLACK =     "\u001b[30m"
RED =       "\u001b[31m"
GREEN =     "\u001b[32m"
YELLOW =    "\u001b[33m"
BLUE =      "\u001b[34m"
MAGENTA =   "\u001b[35m"
CYAN =      "\u001b[36m"
WHITE =     "\u001b[37m"
RESET =     "\u001b[0m"

# Main function
def match(object_file_name, scene_file_name, output_file_name=None, detector='SIFT'):
    # LOAD IMAGES
    img_object = cv2.imread(object_file_name, 0)
    if (img_object is None):
        eprint(RED + "Error:" + RESET + " Unable to read object file")
        return
    img_scene = cv2.imread(scene_file_name, 0)
    if (img_scene is None):
        eprint(RED + "Error:" + RESET + " Unable to read scene file")
        return

    # DETECT AND MATCH FEATURES
    # ORB (Oriented FAST and Rotated BRIEF) Detector
    if (detector == 'ORB'):
        # Initiate ORB detector
        orb = cv2.ORB_create()

        # Find the keypoints and descriptors with ORB
        kp_object, des_object = orb.detectAndCompute(img_object, None)
        kp_scene,  des_scene  = orb.detectAndCompute(img_scene,  None)

        # BFMatcher with default params
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_object, des_scene)

        draw_params = dict(matchColor = (0, 255, 0),        # Green
                           singlePointColor = (255, 0, 0),  # Blue
                           flags = 0)

        img_out = cv2.drawMatches(img_object, kp_object, img_scene, kp_scene, matches, None, **draw_params)

    # SIFT (Scale-Invariant Feature Transform) Detector
    elif (detector == 'SIFT'):
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        # Find the keypoints and descriptors with SIFT
        
        kp_object, des_object = sift.detectAndCompute(img_object, None)
        kp_scene,  des_scene  = sift.detectAndCompute(img_scene,  None)
        
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des_object, des_scene, k=2)
        # Mask out poor matches
        pos_x = 0.0
        pos_y = 0.0
        num_kp = 0
        good_matches = []
        for i, (m,n) in enumerate(matches):
            if m.distance < 0.8 * n.distance: # Distance ratio 0.8, as used in Lowe
                good_matches.append((m, None))

                num_kp += 1
                pos_x += kp_scene[m.trainIdx].pt[0]
                pos_y += kp_scene[m.trainIdx].pt[1]

        if (num_kp > 0):
            pos_x = int(pos_x / float(num_kp))
            pos_y = int(pos_y / float(num_kp))

        draw_params = dict(matchColor = (0, 255, 0),        # Green
                           singlePointColor = (255, 0, 0),  # Blue
                           # matchesMask = matches_mask,
                           flags = 0)

    if (output_file_name is not None):
        img_out = cv2.drawMatchesKnn(img_object, kp_object, img_scene, kp_scene, good_matches, None, **draw_params)
        cv2.imwrite(output_file_name, img_out)

    return (pos_x, pos_y)
    

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', help="Input object image filename", required=True)
    parser.add_argument('-s', '--scene', help="Input scene image filename", required=True)
    parser.add_argument('-o', '--output', help="Output image (with matches shown) filename")

    parser.add_argument('-d', '--detector', choices=['ORB', 'SIFT'], help="Choice of feature detector", default='SIFT')
    
    return parser.parse_args()


if __name__ == '__main__':
    start_time = time.time()

    args = get_args()

    match(args.input, args.scene, args.output, args.detector)

    print("\nTotal time taken: " + str(time.time() - start_time) + " seconds")
    sys.stdout.flush()