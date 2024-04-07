# Structure From Motion Pipeline
## Abstract
This repository contains the following functionality/algorithm implementations for SfM:
| Functionality | Implementation  | Class Name |
| ------------- | ------- | ------- |
| feature detection | OpenCV | view_processor (VP) |
| feature matching | OpenCV | key_tracker (KT) |
| 2D and 3D point tracking | Self | key_tracker |
| fundamental matrix calculation | Self | epipolar_processor (EP) |
| essential matrix calculation | Self | epipolar_processor |
| camera pose extraction from essential matrix | Self | campose_processor (CP) |
| camera pose disambiguation | Self | campose_processor |
| linear triangulation | Self | triangulation_processor (TP) |
| nonlinear triangulation | Self | triangulation_processor |
| linear perspective-n-point estimation | Self | campose_processor |
| nonlinear perspective-n-point estimation | Self | campose_processor |
| bundle adjustment | Self | ba_processor (BAP) |

The pipeline (data flow) is as shown below
![Pipeline](https://github.com/willSapgreen/structure-from-motion/assets/6188375/b268689c-ff1d-4cb2-aefb-818de6150b98)

## Environment
- Python 3.8.17
- Conda environment: environment_sfm.yml

## Commands
Each script has its own unit. Ex. **python3 epipolar_processor.py**.
For the whole pipeline testing, **python3 ba_processor.py**.

## Whole testing result
![ba_processor_unit_test_result](https://github.com/willSapgreen/structure-from-motion/assets/6188375/2b538faa-acea-4580-9ab9-4fc2efc3d3e6)

## Reference
- https://www.cis.upenn.edu/~cis580/Spring2015/Projects/proj2/proj2.pdf
- https://www-users.cse.umn.edu/~hspark/CSci5980/hw5.pdf
- https://www-users.cse.umn.edu/~hspark/csci5563_S2021/hw4.pdf
- [https://cvgl.stanford.edu/teaching/cs231a_winter1415/lecture/lecture3_camera_calibration_notes.pdf](https://cvgl.stanford.edu/teaching/cs231a_winter1415/lecture/)https://cvgl.stanford.edu/teaching/cs231a_winter1415/lecture/
