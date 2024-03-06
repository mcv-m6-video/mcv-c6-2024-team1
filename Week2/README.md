<!-- ⚠️ This README has been generated from the file(s) "blueprint.md" ⚠️-->
[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#week-2)

# ➤ Week 2

During the second week of our project, we focused on object detection and object tracking. We worked on off-the-shelf object detection models, annotation of data for fine-tuning, and fine-tuning models for specific data. We also used different ways of doing K-Fold Cross-validation to evaluate the performance of the models. Additionally, we explored object tracking techniques such as tracking by overlap and Kalman Filter. To evaluate the performance of the tracking methods, we used IDF1 and HOTA scores. Our goal was to explore various techniques to detect and track objects in images and evaluate their effectiveness.



[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#available-tasks)

## ➤ Available tasks

* **Task 1**: Object detection
  * **Task 1.1**: Off-the-shelf
  * **Task 1.2**: Annotation (not executable using the main)
  * **Task 1.3**: Fine-tune to your data
  * **Task 1.4**: K-Fold Cross-validation
* **Task 2**: Object tracking
  * **Task 2.1**: Tracking by overlap
  * **Task 2.2**: Tracking with a Kalman Filter
  * **Task 2.3**: IDF1,HOTA scores


[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#additional-information)

## ➤ Additional information
The annotations for task 1.2 are saved in the task 1.2 folder using the same format as the ones given for S3_C010.



[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#usage)

## ➤ Usage
### Task 1.1
This task can be executed with the following script:
  ```python
  python task1_1.py [--display]
  ```
If display is `True`, then the frames with the corresponding bounding boxes detected will be shown.

### Task 1.3
This task can be executed with the following script:
  ```python
  python task1_3.py
  ```
There are no extra arguments for this task script, given that we decided to set the parameters internally (in the code) while experimenting.

### Task 1.4
This task can be executed with the following script:
  ```python
  python task1_4.py --strategy {A, B, C} 
  ```
The `strategy` flag is used for deciding which strategy to use when using K-Fold evaluation. 

### Task 2.1
This task can be executed with the following script:
  ```python
  python task2_1.py 
  ```
Run `python task2_1.py -h` to check the setable parameters for this script.
Disclaimer: this scripts runs tracking given a json object detection prediction file with YOLO style.

### Task 2.2
Run the script.
  ```python
  python task 2_2.py [--video_path]  [--results_path] [--o_name] [--detections] [--store] [--vizualize] [--thr] [--max_age]
  ```
  Where:
  - `video_path` is the directory of the video from which the detections has been extracted.
  - `results_path` is the follder directory where the outputs will be placed.
  - `o_name` name for the output (JSON and video).
  - `detections` json file with detections.
  - `store` flag to save the video.
  - `vizualize` flag to vizualize the video.
  - `thr` minimum iou to associate a detection with a prediction.
  - `max_age` maximum number of frames before kill a track.

  By default, the script will output a JSON with the tracking information.

### Task 2.3
Convert the outputs of the trackers to the required csv format:
```
python convert_to_trackeval.py
```

Create the folder structure required by TrackEval using the obtained folders
![image](https://github.com/mcv-m6-video/mcv-c6-2024-team1/assets/32550964/a6bc2afc-5f08-4b0b-a3af-f7810999e13f)


Then run the script with the appropiate arguments:
  ```
python task2_3.py --GT_FOLDER .\TrackEval\data\gt\mot_challenge\ --TRACKERS_FOLDER .\TrackEval\data\trackers\mot_challenge\ --BENCHMARK S03aicity --METRICS HOTA Identity --DO_PREPROC False --TRACKERS_TO_EVAL ioutrack
  ```


[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#requirements)

## ➤ Requirements
Install the requirements with the following command:
```python
pip install -r Week2/requirements.txt
```
