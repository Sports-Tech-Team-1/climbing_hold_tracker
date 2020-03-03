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
from collections import deque

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
    ret, frame_bfr = cap.read() 
    ret, frame_cur = cap.read()    
     

    # Gaussian blur parameters
    hold_kernel = (21,21)
    person_kernel = (5,5)
    sigma = 0

    #Threshold parameters
    hold_threshold = 40
    person_threshold = 60
    max_threshold = 255


    #dilate parameters
    dilate_kernel = None
    hold_iterations_Count = 1
    person_iterations_Count = 3

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

    #hold area parameters
    hold_min_area = 2300
    hold_max_area = 5000

    while cap.isOpened():
        # Contour Finding for hold 
        hold_diff = cv2.absdiff (frame_old,frame_cur)
        hold_gray = cv2.cvtColor (hold_diff,cv2.COLOR_BGR2GRAY)
        hold_blur = cv2.GaussianBlur (hold_gray,hold_kernel,sigma)
        _,hold_thresh = cv2.threshold (hold_blur, hold_threshold, max_threshold, cv2.THRESH_BINARY)
        hold_dilated = cv2.dilate (hold_thresh, dilate_kernel, iterations = hold_iterations_Count)
        _,hold_contours,_ = cv2.findContours (hold_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Contour Finding for Person
        person_diff = cv2.absdiff (frame_bfr,frame_cur)
        person_gray = cv2.cvtColor (person_diff,cv2.COLOR_BGR2GRAY)
        person_blur = cv2.GaussianBlur (person_gray,person_kernel,sigma)
        _,person_thresh = cv2.threshold (person_blur, person_threshold, max_threshold, cv2.THRESH_BINARY)
        

        # person_dilated = cv2.dilate (person_thresh, dilate_kernel, iterations = person_iterations_Count)
        # _, person_contours, _ = cv2.findContours (person_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # (x, y, w, h) = cv2.boundingRect (hold_thresh)
        for hold_contour in hold_contours:
            (x, y, w, h) = cv2.boundingRect (hold_contour)
            area = cv2.contourArea (hold_contour)
            if  area > hold_min_area and area < hold_max_area:
                cv2.rectangle (frame_cur, (x, y), (x+w, y+h),blue , thickness )
            
        # for person_contour in person_contours:
        #     (x, y, w, h) = cv2.boundingRect (person_contour)
        #     if cv2.contourArea (person_contour) > 5000:
        #         cv2.rectangle (frame_cur, (x, y), (x+w, y+h), green , thickness )
        #         cv2. putText (frame_cur, "Status: {}".format("Climbing"), font_pos, 
        #                    cv2.FONT_HERSHEY_SIMPLEX, font_size, red, font_thick)
            

        # contours for all objects in current frame
        # cv2. drawContours (frame_1, contours, -1, (0,0,255), 2)

        #  visualization
        cv2.imshow("Camera feed",frame_cur)
        # cv2.imshow("Person feed",hold_thresh)

        # changing the reference frame for hold once in threshold frames
        if frame_Count == frame_count_threshold:
            frame_Count = 0
            frame_old = frame_cur
        else:
            frame_Count += 1
        
        # Current becomes before frame
        frame_bfr = frame_cur
        ret, frame_cur = cap.read()
           
        if cv2.waitKey(40) == 27 or cv2.waitKey(40) == ord("q"):
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


