<!-- ⚠️ This README has been generated from the file(s) "blueprint.md" ⚠️-->
[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#week-2)

# ➤ Week 4

During the fourth week of our project, we focused on estimating the speed of vehicles using visual cues and multi-camera tracking.

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#available-tasks)

## ➤ Available tasks

* **Task 1**: Estimate the speed of vehicles using visual cues
  * **Task 1.1**: Speed estimation
  * **Task 1.2**: Speed estimation with our data
* **Task 2**: Multi-camera tracking



[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#usage)

## ➤ Usage
### Task 1.1

```bash
  python task1_1.py 
```
### Task 1.2

```bash
  python task1_2 
```
### Task 2

```bash
cd Week4/Task2
```

Run mot

TODO

Put training data inside Task2 folder like this:
```bash
Data
└───train
    ├───S01
    │   ├───c001
    │   │  
    │   ├───c002
    │   │  
    │   ├───c003
    │   │  
    │   ├───c004
    │   │  
    │   └───c005
    ├───S03
    │   ├───c010
    │   │  
    │   ├───c011
    │   │  
    │   ├───c012
    │   │  
    │   ├───c013
    │   │  
    │   ├───c014
    │   │  
    │   └───c015
    └───S04
        ├───c016
        ├───c017
        ├───c018
        ├───c019
        ├───c020
        ├───c021
        ├───c022
        ├───c023
        ├───c024
        ├───c025
        ├───c026
        ├───c027
        ├───c028
        ├───c029
        ├───c030
        ├───c031
        ├───c032
        ├───c033
        ├───c034
        ├───c035
        ├───c036
        ├───c037
        ├───c038
        ├───c039
        └───c040
```

```bash
python inverse_projection.py
```
This will save th camera tracklets csv's and the videos for visualization

To compute the IDF1 first concatenate the tracklets files and gt files with:
```bash
python concatenate_csv.py
```

And run trackeval
TODO


[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#requirements)

## ➤ Requirements
Install the requirements with the following command:
```python
pip install -r Week4/requirements.txt
```
