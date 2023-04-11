# End-to-end Autonomous Driving Challenge

InternImage is the baseline model for the CVPR 2023 [End-to-end Autonomous Driving Challenge](https://opendrivelab.com/AD23Challenge.html)

There are 3 tracks that will use InternImage as the baseline model:

1. [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2)

    The primary task of the dataset is scene structure perception and reasoning, which requires the model to recognize the dynamic drivable states of lanes in the surrounding environment. The challenge of this dataset includes not only detecting lane centerlines and traffic elements but also recognizing the attribute of traffic elements and topology relationships on detected objects. 

2. [Online HD Map Construction](https://github.com/Tsinghua-MARS-Lab/Online-HD-Map-Construction-CVPR2023)

    Constructing HD maps is a central component of autonomous driving. However, traditional mapping pipelines require a vast amount of human efforts in annotating and maintaining the map, which limits its scalability. Online HD map construction task aims to dynamically construct the local semantic map based on onboard sensor observations. Compared to lane detection, our constructed HD map provides more semantics information of multiple categories. Vectorized polyline representation are adopted to deal with complicated and even irregular road structures.

3. [3D Occupancy Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction)

    Understanding the 3D surroundings including the background stuffs and foreground objects is important for autonomous driving. In the traditional 3D object detection task, a foreground object is represented by the 3D bounding box. However, the geometrical shape of the object is complex, which can not be represented by a simple 3D box, and the perception of the background is absent. The goal of this task is to predict the 3D occupancy of the scene. In this task, we provide a large-scale occupancy benchmark based on the nuScenes dataset. The benchmark is a voxelized representation of the 3D space, and the occupancy state and semantics of the voxel in 3D space are jointly estimated in this task. The complexity of this task lies in the dense prediction of 3D space given the surround-view image.
