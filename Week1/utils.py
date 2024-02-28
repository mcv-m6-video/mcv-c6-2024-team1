import cv2
import numpy as np
import xml.etree.ElementTree as ET
import random

import numpy as np
import cv2
import datetime
from lxml import etree

import matplotlib.pyplot as plt
import rembg
from PIL import Image
from tqdm import tqdm
import pickle
import os
import argparse
import glob
import random

def binaryMaskIOU(mask1, mask2):
    mask1_area = np.count_nonzero(mask1 == 255)
    mask2_area = np.count_nonzero(mask2 == 255)
    intersection = np.count_nonzero(np.logical_and(mask1 == 255, mask2 == 255))
    union = mask1_area+mask2_area-intersection
    if union == 0: # Evitar dividir entre 0
        return 0
    iou = intersection/(union)
    return iou

# Read the XML file of week 1 GT annotations
def readXMLtoAnnotation(annotationFile, remParked = False):
    """
    Read XML file of annotations and parse it to annotation dictionary.

    Parameters
    ----------
    annotationFile : str
        Path to the XML annotation file.
    remParked : bool, optional
        Whether to remove parked objects from the annotations, by default False.

    Returns
    -------
    dict, list
        Dictionary containing annotations and list of image ids.
    """
    # Read XML
    file = ET.parse(annotationFile)
    root = file.getroot()
    annotations = {}

    # Find objects
    for child in root:
        if child.tag == "track":
            # Get class
            className = child.attrib["label"]
            for obj in child:
                if className == "car":
                    objParked = obj[0].text
                    # Do not store if it is parked and we want to remove parked objects
                    if objParked == "true" and remParked:
                        continue
                frame = obj.attrib["frame"]
                xtl = float(obj.attrib["xtl"])
                ytl = float(obj.attrib["ytl"])
                xbr = float(obj.attrib["xbr"])
                ybr = float(obj.attrib["ybr"])
                bbox = [xtl, ytl, xbr, ybr]
                if frame in annotations:
                    annotations[frame].append({"name": className, "bbox": bbox})
                else:
                    annotations[frame] = [{"name": className, "bbox": bbox}]

    return annotations

# Remove the annotations until a certain number of frame
def removeFirstAnnotations(stopFrame, annots):
    newAnnots = {}

    # Store only next frame annotations
    for frame in annots.keys():
        num = int(frame)
        if num > stopFrame:
            newAnnots[frame] = annots[frame]
            
    return newAnnots

def read_annotations(annotations_path: str):
    """
    Function to read the GT annotations from ai_challenge_s03_c010-full_annotation.xml

    At the moment we will only check that the track is for "car" and has "parked" as false
    and we will save the bounding box attributes from the 'box' element.
    """
    tree = etree.parse(annotations_path)
    root = tree.getroot()
    car_boxes = {}

    for track in root.xpath(".//track[@label='car']"):
        track_id = track.get("id")
        for box in track.xpath(".//box"):
            parked_attribute = box.find(".//attribute[@name='parked']")
            if parked_attribute is not None and parked_attribute.text == 'false':
                frame = box.get("frame")
                box_attributes = {
                    "xtl": box.get("xtl"),
                    "ytl": box.get("ytl"),
                    "xbr": box.get("xbr"),
                    "ybr": box.get("ybr"),
                    # in the future we will need more attributes
                }
                if frame in car_boxes:
                    car_boxes[frame].append(box_attributes)
                else:
                    car_boxes[frame] = [box_attributes]

    return car_boxes
# Read txt detection lines to annot
def readTXTtoDet(txtPath):
    """
    Read detections from a text file and parse them into image ids, confidences, and bounding boxes.

    Parameters
    ----------
    txtPath : str
        Path to the text file containing detections.

    Returns
    -------
    tuple
        Tuple containing image ids, confidences, and bounding boxes.
    """
    # Read file
    file = open(txtPath, 'r')
    lines = file.readlines()
    # Init values
    imageIds = []
    confs = []
    BB = np.zeros((0,4))
    # Insert every detection
    for line in lines:
        #frame,-1,left,top,width,height,conf,-1,-1,-1
        splitLine = line.split(",")
        # Frame
        imageIds.append(str(int(splitLine[0])-1))
        # Conf
        confs.append(float(splitLine[6]))
        # BBox
        left = float(splitLine[2])
        top = float(splitLine[3])
        width = float(splitLine[4])
        height = float(splitLine[5])
        xtl = left
        ytl = top
        xbr = left + width - 1
        ybr = top + height - 1
        BB = np.vstack((BB, np.array([xtl,ytl,xbr,ybr])))
    
    file.close()
    
    return (imageIds, np.array(confs), BB)


# Parse from annotations format to detection format
def annoToDetecFormat(annot, className):
    """
    Convert annotations to detection format for a specific class.

    Parameters
    ----------
    annot : dict
        Dictionary containing annotations.
    className : str
        Name of the class for which annotations should be converted.

    Returns
    -------
    list, np.ndarray
        List of image ids and array of bounding boxes in detection format.
    """
    
    imageIds = []
    BB = np.zeros((0,4))
    
    for imageId in annot.keys():
        for obj in annot[imageId]:
            
            if obj["name"] == className:
                imageIds.append(imageId)
                BB = np.vstack((BB, obj["bbox"]))
    
    return imageIds, BB

def read_video(vid_path: str):
    vid = cv2.VideoCapture(vid_path)
    frames = []
    color_frames = []
    while True:
        ret, frame = vid.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            color_frames.append(frame_rgb)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame_gray)
        else:
            break
    vid.release()
    return np.array(frames), np.array(color_frames)

# Draw detection and annotation boxes in image
def drawBoxes(img, det, annot, colorDet, colorAnnot):
    """
    Draw detection and annotation boxes on an image.

    Parameters
    ----------
    img : np.ndarray
        Image on which boxes will be drawn.
    det : np.ndarray
        Array of bounding boxes for detections.
    annot : list
        List of dictionaries containing annotation information.
    colorDet : tuple
        Color for drawing detection boxes.
    colorAnnot : tuple
        Color for drawing annotation boxes.

    Returns
    -------
    np.ndarray
        Image with drawn boxes.
    """
    img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Draw annotations
    for obj in annot:
            # Draw box
            bbox = obj["bbox"]
            img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), 
                               (int(bbox[2]), int(bbox[3])), colorAnnot, 3)
    
    # Draw detections
    for i in range(det.shape[0]):
        # Draw box
        bbox = det[i,:]
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), 
                           (int(bbox[2]), int(bbox[3])), colorDet, 3)
    
    return img


def makeVideo(images, videoName, fps = 10):
    """
    Create a video from a list of images.

    Parameters
    ----------
    images : list
        List of images to create the video from.
    videoName : str
        Name of the output video file.
    fps : int, optional
        Frames per second for the output video, by default 10.

    Returns
    -------
    None
    """
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(videoName, fourcc, fps, (images[0].shape[1], images[0].shape[0]), True)

    for i in range(len(images)):
        out.write(images[i].astype(np.uint8))

    out.release()

def randomFrame(videoPath):
    """
    Read a random frame from a video.

    Parameters
    ----------
    videoPath : str
        video path.

    Returns
    -------
    image : numpy array
        random frame.
    randomFrameNumber : int
        random frame number.

    """
    vidcap = cv2.VideoCapture(videoPath)
    # get total number of frames
    totalFrames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    randomFrameNumber=random.randint(0, totalFrames)
    # set frame position
    vidcap.set(cv2.CAP_PROP_POS_FRAMES,randomFrameNumber)
    success, image = vidcap.read()
    
    
    return image, randomFrameNumber

def split_frames(frames):
    """
    Returns 25% and 75% split partition of frames.
    """
    return frames[:int(frames.shape[0] * 0.25), :, :], frames[int(frames.shape[0] * 0.25):, :, :]

def detect_obj(frame_idx, gray_frame, color_frame, gt):
    gray_frame = (gray_frame * 255).astype(np.uint8)

    #plt.imsave(f'./pruebas_1_1/before_{frame_idx + inc}.jpg', gray_frame, cmap='gray')

    # Opening
    # kernel = np.ones((3,3),np.uint8)
    # gray_frame = cv2.morphologyEx(gray_frame, cv2.MORPH_OPEN, kernel)

    # Connected components
    contours, _ = cv2.findContours(
        gray_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    predictions = []
    pred_mask_list = []
    height = gray_frame.shape[0]
    width = gray_frame.shape[1]
    thr = 3000
    pred_mask = np.zeros(gray_frame.shape, dtype="uint8")
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > thr:
            contour = contour[:, 0, :]
            xmin = int(np.min(contour[:, 0]))
            ymin = int(np.min(contour[:, 1]))
            xmax = int(np.max(contour[:, 0]))
            ymax = int(np.max(contour[:, 1]))
            if (xmax - xmin) < width * 0.4 and (ymax - ymin) < height * 0.4:
                pred = {"bbox": [xmin, ymin, xmax, ymax]}
                predictions.append(pred)
                cv2.rectangle(color_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                cv2.rectangle(pred_mask, (xmin, ymin), (xmax, ymax), 255, -1)
                pred_mask_list.append(pred_mask)
                
    output = np.zeros(gray_frame.shape, dtype="uint8")

    
    # Paint the GT Bounding boxes
    gt_mask_list = []

    real_frame_idx = str(frame_idx)
    if real_frame_idx in gt:
        for box in gt[real_frame_idx]:
            gt_mask = np.zeros(gray_frame.shape, dtype="uint8") # gt
            xtl = int(float(box['xtl']))
            ytl = int(float(box['ytl']))
            xbr = int(float(box['xbr']))
            ybr = int(float(box['ybr']))
            
            cv2.rectangle(color_frame, (xtl, ytl), (xbr, ybr), (0, 0, 255), 3)
            cv2.rectangle(gt_mask, (xtl, ytl), (xbr, ybr), 255, -1)

            #plt.imsave(f'./pruebas_1_1/gt_mask_{len(gt_mask_list)}.jpg', gt_mask, cmap="gray")

            gt_mask_list.append(gt_mask)
    
    # Compute metrics
    precision = compute_ap(gt_mask_list, pred_mask_list)
    recall = compute_ap(pred_mask_list, gt_mask_list)

    return precision, recall

def compute_ap(gt_boxes, pred_boxes):
    # Initialize variables
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    gt_matched = np.zeros(len(gt_boxes))

    # Iterate over the predicted boxes
    for i, pred_box in enumerate(pred_boxes):
        ious = [binaryMaskIOU(pred_box, gt_box) for gt_box in gt_boxes]
        if len(ious) == 0:
            fp[i] = 1
            continue
        max_iou = max(ious)
        max_iou_idx = ious.index(max_iou)

        if max_iou >= 0.5 and not gt_matched[max_iou_idx]:
            tp[i] = 1
            gt_matched[max_iou_idx] = 1
        else:
            fp[i] = 1

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / len(gt_boxes)
    # if len(gt_boxes) > 0:
    #     recall = tp / len(gt_boxes)
    # else:
    #     recall = 0
    precision = tp / (tp + fp)

    # Generate graph with the 11-point interpolated precision-recall curve
    recall_interp = np.linspace(0, 1, 11)
    precision_interp = np.zeros(11)
    for i, r in enumerate(recall_interp):
        array_precision = precision[recall >= r]
        if len(array_precision) == 0:
            precision_interp[i] = 0
        else:
            precision_interp[i] = max(precision[recall >= r])

    ap = np.mean(precision_interp)
    return ap