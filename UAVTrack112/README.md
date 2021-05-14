# [UAVTrack112]

## Sequence number
This benchmark is collected by DJI Mavic Air2 with 112 sequences. The structure of the files is as follows:

```
UAVTrack112

    --data_seq \ the sequence of images
          ...

    --anno \ the ground truth of each frame 
          ...

    --att \ the attributes of each sequence
          ...

```

**Note:** The format of ground truth adopts the common way (left-top x-coordinate, left-top y-coordinate, width, height) to represent the ground truth bounding box.


**Note:** This aerial tracking benchmark involves 11 most common challenges, i.e., fast motion, low resolution, long-term tracking, 



