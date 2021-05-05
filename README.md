# Capstone Project
## Pedestrian Group Detection 
[![DOI](https://zenodo.org/badge/353060555.svg)](https://zenodo.org/badge/latestdoi/353060555)

[GitHub.io](https://github.com/huzzie/huzzie.github.io-capstone)
# Abstract
The edge-computing, a distributed computing system that processes complex computation and brings this computation to the source of the data, is a growing innovation technique in artificial intelligence and IoT (Internet of Things). Using the YOLO on NVIDIA Jetson-Nano, pedestrians in the Washington, DC area are detected, and the methods for pedestrian feature detection proposed based on YOLO and K-means clusters. YOLO model can detect more than 9000 objects with high accuracy score, but group detection remains as a challenging task. This paper will introduce four methods to detect groups, and these analyses will provide insights to city developers to find out business potentials and bring insights into the communities. In the research, the algorithm based on YOLOv4 will detect the existing real-time pedestrian and K-means clustering will be used to measure and evaluate group units.

# Keywords
Object Detection, Motion Detection, K-means, DBSCAN, YOLO

# Data Collection
Download the [Yolo Model](https://github.com/theAIGuysCode/yolov4-deepsort) and use the object_track.py under the [codes](https://github.com/huzzie/Capstone_Project/tree/main/codes) folder to transform the pedestrian information into CSV format. GPU system is highly recommmended. 

# Object Trajectories
![my_video](/.video_result/crowd.mp4)
# References
[Deep Sort](https://github.com/nwojke/deep_sort)
[Yolo Model](https://github.com/theAIGuysCode/yolov4-deepsort)
