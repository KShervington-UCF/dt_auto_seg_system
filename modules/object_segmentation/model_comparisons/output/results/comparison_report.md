# Model Performance Comparison Report

Date: 2025-06-17 20:33:08
Number of test images: 5

## Summary Results

| Model       |   Avg. Inference Time (s) |   Avg. Objects Detected |   Avg. Confidence |
|:------------|--------------------------:|------------------------:|------------------:|
| yolo11n-seg |                    0.1397 |                    14.6 |            0.5944 |
| yolo11s-seg |                    0.1403 |                    15.2 |            0.6352 |
| yolo11m-seg |                    0.1587 |                    15.6 |            0.6493 |
| yolo11l-seg |                    0.167  |                    17   |            0.6369 |
| yolo11x-seg |                    0.2768 |                    16.6 |            0.65   |
| detectron2  |                    0.5162 |                    17   |            0.8728 |

## Per-Image Results

### img_1.png

| Model       |   Avg. Inference Time (s) |   Avg. Objects Detected |   Avg. Confidence |
|:------------|--------------------------:|------------------------:|------------------:|
| yolo11n-seg |                    0.3234 |                      18 |            0.6485 |
| yolo11s-seg |                    0.348  |                      19 |            0.6918 |
| yolo11m-seg |                    0.4447 |                      19 |            0.6942 |
| yolo11l-seg |                    0.5616 |                      19 |            0.7227 |
| yolo11x-seg |                    0.9167 |                      18 |            0.7272 |
| detectron2  |                    1.045  |                      19 |            0.8827 |

### img_2.png

| Model       |   Avg. Inference Time (s) |   Avg. Objects Detected |   Avg. Confidence |
|:------------|--------------------------:|------------------------:|------------------:|
| yolo11s-seg |                    0.0693 |                      19 |            0.671  |
| yolo11n-seg |                    0.0702 |                      19 |            0.6307 |
| yolo11m-seg |                    0.0867 |                      19 |            0.6802 |
| yolo11l-seg |                    0.0888 |                      21 |            0.6617 |
| yolo11x-seg |                    0.091  |                      20 |            0.6422 |
| detectron2  |                    0.4876 |                      19 |            0.8966 |

### img_3.jpg

| Model       |   Avg. Inference Time (s) |   Avg. Objects Detected |   Avg. Confidence |
|:------------|--------------------------:|------------------------:|------------------:|
| yolo11l-seg |                    0.0591 |                      18 |            0.6311 |
| yolo11m-seg |                    0.0783 |                      18 |            0.633  |
| yolo11s-seg |                    0.0936 |                      17 |            0.605  |
| yolo11n-seg |                    0.104  |                      20 |            0.5047 |
| yolo11x-seg |                    0.1295 |                      19 |            0.6246 |
| detectron2  |                    0.404  |                      22 |            0.8412 |

### img_4.jpg

| Model       |   Avg. Inference Time (s) |   Avg. Objects Detected |   Avg. Confidence |
|:------------|--------------------------:|------------------------:|------------------:|
| yolo11l-seg |                    0.0647 |                      14 |            0.5245 |
| yolo11m-seg |                    0.0929 |                      12 |            0.5456 |
| yolo11n-seg |                    0.0994 |                       8 |            0.5945 |
| yolo11s-seg |                    0.1004 |                      10 |            0.5861 |
| yolo11x-seg |                    0.1203 |                      12 |            0.577  |
| detectron2  |                    0.3472 |                      13 |            0.8533 |

### img_5.jpg

| Model       |   Avg. Inference Time (s) |   Avg. Objects Detected |   Avg. Confidence |
|:------------|--------------------------:|------------------------:|------------------:|
| yolo11l-seg |                    0.0606 |                      13 |            0.6004 |
| yolo11s-seg |                    0.0901 |                      11 |            0.5672 |
| yolo11m-seg |                    0.0909 |                      10 |            0.6588 |
| yolo11n-seg |                    0.1015 |                       8 |            0.6106 |
| yolo11x-seg |                    0.1266 |                      14 |            0.6591 |
| detectron2  |                    0.2973 |                      12 |            0.8986 |

## Visualization Paths

Visualization images are saved in the following directories:

- img_1.png: D:\Projects\dt_auto_seg_system\modules\object_segmentation\model_comparisons\output\visualizations\img_1
- img_2.png: D:\Projects\dt_auto_seg_system\modules\object_segmentation\model_comparisons\output\visualizations\img_2
- img_3.jpg: D:\Projects\dt_auto_seg_system\modules\object_segmentation\model_comparisons\output\visualizations\img_3
- img_4.jpg: D:\Projects\dt_auto_seg_system\modules\object_segmentation\model_comparisons\output\visualizations\img_4
- img_5.jpg: D:\Projects\dt_auto_seg_system\modules\object_segmentation\model_comparisons\output\visualizations\img_5
