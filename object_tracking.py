# object_tracking.py
# Human and hold detector
#
# Author: Nanjayan (nnanjaya@andrew.cmu.edu)
# Date: Mar. 2, 2020
#
# Description: 
# Given a video stream, it attempts to locate the object i.e hold and the human within the scene.
# A green box is drawn around the human and blue box is drawn over the hold
# Matches can be outputted to an output image for visualization.
# 
# Currently,it supports the Contour based detectors
#
# Usage: object_tracking.py [-h] -v VIDEO
# optional arguments:
#   -h, --help                                                  show this help message and exit
#   -vd VIDEO_DIRECTORY, --video_directory VIDEO_DIRECTORY      Input Video directory name
#
#
# References:
# https://www.youtube.com/watch?v=MkcUgPhOlP8&list=PLS1QulWo1RIa7D1O6skqDQ-JZ1GGHKK-K&index=28
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



def object_tracking (video):

    cap = cv2.VideoCapture(video)

    ret, frame_old = cap.read()
    ret, frame_cur = cap.read()    

    # Gaussian blur parameters
    kernel = (21,21)
    sigma = 0

    #Threshold parameters
    threshold = 20
    max_threshold = 255

    #dilate parameters
    dilate_kernel = None
    iterations_Count = 1

    #Text and color properties 
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)
    thickness = 2
    font_size = 1
    font_thick = 5 
    font_pos = (50,50)

    # change in replacing old frame
    frame_Count = 0
    frame_count_threshold = 50

    while cap.isOpened():

        diff = cv2.absdiff (frame_old,frame_cur)
        gray = cv2.cvtColor (diff,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur (gray,kernel,sigma)
        _,thresh = cv2.threshold (blur, threshold, max_threshold, cv2.THRESH_BINARY)
        dilated = cv2.dilate (thresh, dilate_kernel, iterations = iterations_Count)
        _, contours, _ = cv2.findContours (dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect (contour)

            if cv2.contourArea (contour) > 1300 and cv2.contourArea (contour) < 5000:
                cv2.rectangle (frame_cur, (x, y), (x+w, y+h),blue , thickness )
            # elif cv2.contourArea (contour) > 5000:
            #     cv2.rectangle (frame_cur, (x, y), (x+w, y+h),green , thickness )
            #     cv2. putText (frame_cur, "Status: {}".format("Climbing"), font_pos, 
            #                cv2.FONT_HERSHEY_SIMPLEX, font_size, red, font_thick)
            

        # contours for all objects
        # cv2. drawContours (frame_1, contours, -1, (0,0,255), 2)

        #  visualization
        cv2.imshow("Camera feed",frame_cur)

        frame_Count += 1

        if frame_Count == frame_count_threshold:
            frame_Count = 0
            frame_old = frame_cur

        ret, frame_cur = cap.read()

           
        if cv2.waitKey(40) == 27 or cv2.waitKey(40) == 'q':
            break
    
    cv2.destroyAllWindows()
    cap.release()



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-vd', '--video_directory', help="Input video directory name", required=True)
    
    return parser.parse_args()

if __name__ == '__main__':
    start_time = time.time()

    args = get_args()

    object_tracking (args.video_directory)

    print("\nTotal time taken: " + str(time.time() - start_time) + " seconds")
    sys.stdout.flush()


