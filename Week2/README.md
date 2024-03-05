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
  python task1_4.py --strategy {A, B, C} --fold-index {0, 1, 2, 3}
  ```
The `strategy` and `fold-index` flags are used for deciding which strategy to use when using K-Fold evaluation. If the `strategy` decided is `B`, then the parameter `fold-index` needs to be set.

### Task 2.1
  ```python
  python 
  ```
### Task 2.2
  ```python
  python 
  ```
### Task 2.3
  ```python
  python 
  ```


[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#requirements)

## ➤ Requirements

