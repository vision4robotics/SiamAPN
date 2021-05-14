# [UAVTrack112]

## Sequence number
This benchmark is collected by DJI Mavic Air2 with 112 sequences. The structure of the files is as follows:

```
UAVTrack112

    --data_seq \ the sequence of images
          ...

    --anno \ the ground truth of each frame 
          ...

    --att \ the 13 attributes of each sequence
          ...

```

**Note:** The format of ground truth adopts the common way (left-top x-coordinate, left-top y-coordinate, width, height) to represent the ground truth bounding box.


**Note:** The format of attributes is ordered as Fast Motion, Low Resolution, Long-term Tracking, Aspect Ratio Change, Scale Variation, Partial Occlusion, Full Occlusion, Camera Motion, Out-of-View, Illumination Variation, Low Illumination, Viewpoint Change, Similar Object.

 

Fast Motion: motion of the ground truth bounding box is larger than
20 pixels between two consecutive frames.

Low Resolution: at least one ground truth bounding box has less than 400 pixels.

Long-term Tracking: the sequences with more than 1000 frames.

Aspect Ratio Change: the fraction of ground truth aspect ratio in the first frame
and at least one subsequent frame is outside the range [0.5, 2].

Scale Variation: the ratio of initial and at least one subsequent bounding box
is outside the range [0.5, 2].

Partial Occlusion: the target is partially occluded.

Full Occlusion: the target is fully occluded.

Camera Motion: abrupt motion of the camera.

Out-of-View: some portion of the target leaves the view.

Illumination Variation: the illumination of the target changes significantly.

Low Illumination: the illumination of surroundings is insufficient.

Viewpoint Change: viewpoint affects target appearance significantly.

Similar Object: there are objects of similar appearance near the target.