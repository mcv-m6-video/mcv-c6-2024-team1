<!-- ⚠️ This README has been generated from the file(s) "blueprint.md" ⚠️-->
[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#week-2)

# ➤ Week 3

During the third week of our project, we focused on estimating the optical flow of a video sequence, estimating the optical flow and trying to improve an object tracking algorithm and finally we provided results on data from the CVPR AI City Challenge.



[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#available-tasks)

## ➤ Available tasks

* **Task 1**: Optical Flow
  * **Task 1.1**: Optical Flow with Block Matching
  * **Task 1.2**: Off-the-shelf Optical Flow
  * **Task 1.3**: Object Tracking with Optical Flow
* **Task 2**: Multi-target single-camera (MTSC) tracking
  * **Task 2.1**: SEQ 01
  * **Task 2.2**: SEQ 03
  * **Task 2.3**: SEQ 04



[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#usage)

## ➤ Usage
### Task 1.1

### Task 1.2
The main objective of the task can be performed using script `task1_2.py`, which can be run the following way:
```bash
  python task1_2 [--sequence SEQUENCE]
```
Where `sequence` is the sequence that we want to compute the optical flow from. By default, the sequence is `000045`.

Additionally, there is another script, that generates a video of the optical flow generated from a given input video. It can be run the following way:
```bash
  python optical_flow_video.py [--flow-method FLOW_METHOD] [--input-path INPUT_PATH] [--output-path OUTPUT_PATH]
```
Where:
  - `--flow-method` is the method used (either `pyflow` or `farneback`)
  - `--input-path` is the input path for the given video
  - `--output-path` is the output path for the generated video

### Task 1.3

### Task 2.1

### Task 2.2

### Task 2.3



[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#requirements)

## ➤ Requirements
Install the requirements with the following command:
```python
pip install -r Week3/requirements.txt
```
