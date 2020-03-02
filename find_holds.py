# find_holds.py
# Rock Climbing Hold Finder
#
# Author: James Zhang (jameszha@andrew.cmu.edu)
# Date: Mar. 1, 2020
#
# Description: 
# Given a directory containing images of holds and an image of the wall, find_holds attempts to locate each hold.
# A list of positions of each of the holds is returned
# Holds can be labeled and outputted to an image for visualization
#
#
# Usage: find_holds.py [-h] -hd HOLDS_DIRECTORY -w WALL [-o OUTPUT]
# optional arguments:
#   -h, --help                                                  show this help message and exit
#   -hd HOLDS_DIRECTORY, --holds_directory HOLDS_DIRECTORY      Holds directory name
#   -w WALL, --wall WALL                                        Wall image filename
#   -o OUTPUT, --output OUTPUT                                  Output image (with holds labeled) filename 
#
#
# FAIR USE NOTICE: This file contains copyrighted material the use of which has not been specifically authorized by the copyright
# owner. This material is to be used for research and educational purposes only. We consider use of any such copyrighted material 
# to be fair use per section 107 of the US Copyright Law.
#

# Support libraries
import argparse
import os
import sys
import time
from tqdm import tqdm

# Image processing libraries
# $ pip3 install opencv-contrib-python==3.4.2.16
import numpy as np 
import cv2

from match import match

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

# Scaling factors for drawing onto image
CIRCLE_SCALING = 0.05
LINE_SCALING = 0.003
TEXT_SCALING = 0.002

# Given an image and a position of a hold, circle the hold and label with the hold's index, i
def label_img(img, position, i):
    img_width = np.size(img, 1)

    color = (0, 0, 255) # Red

    circle_size = max(int(img_width*CIRCLE_SCALING), 1)     # Clamp to > 0
    circle_thickness = max(int(img_width*LINE_SCALING), 1)  # Clamp to > 0
    font_size = max(int(img_width*TEXT_SCALING), 1)         # Clamp to > 0
    font_thickness = max(int(img_width*TEXT_SCALING), 1)    # Clamp to > 0

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.circle(img, position, circle_size, color, circle_thickness)
    cv2.putText(img, str(i), position, font, font_size, color, font_thickness) 

# Main function
def find_holds(holds_directory_name, wall_file_name, output_file_name):

    # Get all holds file names
    holds_file_names = [os.path.join(holds_directory_name, f) for f in os.listdir(holds_directory_name) if os.path.isfile(os.path.join(holds_directory_name, f))]

    # Load wall image
    img = cv2.imread(wall_file_name)
    if (img is None):
        eprint(RED + "Error:" + RESET + " Unable to read wall image file")
        return

    # Find all holds
    hold_positions = []
    for i, hold_file_name in enumerate(tqdm(sorted(holds_file_names))):
        position = match(hold_file_name, wall_file_name, detector='SIFT')
        hold_positions.append(position)

        label_img(img, position, i)

    if (output_file_name is not None):
        cv2.imwrite(output_file_name, img)

    return hold_positions


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-hd', '--holds_directory', help="Holds directory name", required=True)
    parser.add_argument('-w', '--wall', help="Wall image filename", required=True)
    parser.add_argument('-o', '--output', help="Output image (with holds labeled) filename")
    
    return parser.parse_args()


if __name__ == '__main__':
    start_time = time.time()

    args = get_args()

    find_holds(args.holds_directory, args.wall, args.output)

    print("\nTotal time taken: " + str(time.time() - start_time) + " seconds")
    sys.stdout.flush()