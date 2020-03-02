# Climbing Hold Tracker

## Introduction
Rock climbing has been rapidly gaining popularity in the past few years. This project seeks to augment this growing sport using the latest in computer vision and machine learning technologies, improving the climbing experience for beginners and seasoned veterans alike.

Currently, only hold-finding on a still image is supported. The Climbing Hold Tracker will track designated holds along a climbing route in a video feed. The Climbing Hold Tracker will serve as the foundation for the more extensive Virtual Climbing Coach.

## Table of contents
<!--ts-->
   * [Dependencies](#dependencies)
   * [Running the code](#running)
   * [TODO](#todo)
<!--te-->

<a name="dependencies"></a>
## Dependencies
This code has been tested using the following dependencies. More updated versions may be usable, but are not guaranteed to work. In particular, versions of opencv-contrib-python above 3.4.2.17 are not compiled with SIFT.
```
Python 3.5.2
```
Python Libraries:
```
opencv-contrib-python 3.4.2.16
tqdm 4.43.0
numpy 1.16.1
```
Example Python library installation:
```
pip3 install opencv-contrib-python==3.4.2.16
```

<a name="running"></a>
## Running the code
- Obtain images of a full climbing wall
- Obtain images of each climbing hold to be tracked
- Run the hold finder using
  ```
  $ python3 find_holds.py -hd <hold directory> -w <wall image> -o <output image>
  ```
  For example, 
  ```
  python3 find_holds.py -hd images/example1/holds -w images/example1/wall.png -o images/example1/out.png
  ```
  
<a name="todo"></a>
## TODO
- Extend hold tracking to operate on video stream
- Improve performance of SIFT keypoint computation
- Verify accuracy versus angle discrepancy of hold and wall images
- Integrate with climber body tracking and move prediction
