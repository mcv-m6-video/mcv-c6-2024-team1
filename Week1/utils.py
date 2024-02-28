import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from lxml import etree

from metrics import AP


# Read the XML file of week 1 GT annotations
def readXMLtoAnnotation(annotationFile, remParked=False):
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
    file = open(txtPath, "r")
    lines = file.readlines()
    # Init values
    imageIds = []
    confs = []
    BB = np.zeros((0, 4))
    # Insert every detection
    for line in lines:
        # frame,-1,left,top,width,height,conf,-1,-1,-1
        splitLine = line.split(",")
        # Frame
        imageIds.append(str(int(splitLine[0]) - 1))
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
        BB = np.vstack((BB, np.array([xtl, ytl, xbr, ybr])))

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
    BB = np.zeros((0, 4))

    for imageId in annot.keys():
        for obj in annot[imageId]:

            if obj["name"] == className:
                imageIds.append(imageId)
                BB = np.vstack((BB, obj["bbox"]))

    return imageIds, BB


def readVideo(vid_path: str):
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
        img = cv2.rectangle(
            img,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            colorAnnot,
            3,
        )

    # Draw detections
    for i in range(det.shape[0]):
        # Draw box
        bbox = det[i, :]
        img = cv2.rectangle(
            img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), colorDet, 3
        )

    return img


def makeVideo(images, videoName, fps=10):
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
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        videoName, fourcc, fps, (images[0].shape[1], images[0].shape[0]), True
    )

    for i in range(len(images)):
        out.write(images[i].astype(np.uint8))

    out.release()


def compute_detections_and_metrics(frame_number, binary_frame, ground_truth):
    # Convert binary frame to 8-bit image
    binary_frame = (binary_frame * 255).astype(np.uint8)

    # Detect contours in the binary frame
    contours, _ = cv2.findContours(
        binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detected_objects = []
    detection_masks = []
    frame_height, frame_width = binary_frame.shape
    min_area_threshold = 3000
    detection_mask = np.zeros(binary_frame.shape, dtype="uint8")

    # Process each detected contour
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > min_area_threshold:
            contour = contour[:, 0, :]
            min_x, min_y = np.min(contour, axis=0).astype(int)
            max_x, max_y = np.max(contour, axis=0).astype(int)

            # Filter out overly large detections
            if (max_x - min_x) < frame_width * 0.4 and (
                max_y - min_y
            ) < frame_height * 0.4:
                detected_object = {"bbox": [min_x, min_y, max_x, max_y]}
                detected_objects.append(detected_object)
                cv2.rectangle(detection_mask, (min_x, min_y), (max_x, max_y), 255, -1)
                detection_masks.append(detection_mask.copy())

    # Create masks for ground truth bounding boxes
    gt_masks = []
    frame_key = str(frame_number)
    if frame_key in ground_truth:
        for bbox in ground_truth[frame_key]:
            gt_mask = np.zeros(binary_frame.shape, dtype="uint8")
            xtl, ytl = int(float(bbox["xtl"])), int(float(bbox["ytl"]))
            xbr, ybr = int(float(bbox["xbr"])), int(float(bbox["ybr"]))
            cv2.rectangle(gt_mask, (xtl, ytl), (xbr, ybr), 255, -1)
            gt_masks.append(gt_mask)

    # Calculate metrics
    precision_metric = AP(gt_masks, detection_masks)
    recall_metric = AP(detection_masks, gt_masks)

    return precision_metric, recall_metric
