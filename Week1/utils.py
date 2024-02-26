import cv2
import numpy as np
import xml.etree.ElementTree as ET
import random

# Read the XML file of week 1 GT annotations
def readXMLtoAnnotation(annotationFile, remParked = False):
    # Read XML
    file = ET.parse(annotationFile)
    root = file.getroot()
    
    annotations = {}
    image_ids = []
    # Find objects
    for child in root:
        if child.tag == "track":
            # Get class
            className = child.attrib["label"]
            #if className != classObj:
            #    continue
            for obj in child:
                if className == "car":
                    objParked = obj[0].text
                    # Do not store if it is parked and we want to remove parked objects
                    if objParked=="true" and remParked:
                        continue
                frame = obj.attrib["frame"]
                xtl = float(obj.attrib["xtl"])
                ytl = float(obj.attrib["ytl"])
                xbr = float(obj.attrib["xbr"])
                ybr = float(obj.attrib["ybr"])
                bbox = [xtl, ytl, xbr, ybr]
                if frame in image_ids:
                    annotations[frame].append({"name": className, "bbox": bbox})
                else:
                    image_ids.append(frame)
                    annotations[frame] = [{"name": className, "bbox": bbox}]
    
    
    return annotations, image_ids

def removeFirstAnnotations(stopFrame, annots, imageIds):
    """
    This function removes the annotations until a certain number of frame

    Parameters
    ----------
    stopFrame : int
        Until what number of frame remove the annotations.
    annots : dict
        Dictionary of annotations.
    imageIds : list
        List of frames of annotations.

    Returns
    -------
    newAnnots : dict
        Annotations with the removed frames.
    newImageIds : list
        Annotation frame ids with removed frames.

    """
    newAnnots = {}
    newImageIds = []
    
    # Store only next frame annotations
    for frameNum in imageIds:
        num = int(frameNum)
        if num > stopFrame:
            newImageIds.append(frameNum)
            newAnnots[frameNum] = annots[frameNum]
            
    return newAnnots, newImageIds

# Read txt detection lines to annot
def readTXTtoDet(txtPath):
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
    
    imageIds = []
    BB = np.zeros((0,4))
    
    for imageId in annot.keys():
        for obj in annot[imageId]:
            
            if obj["name"] == className:
                imageIds.append(imageId)
                BB = np.vstack((BB, obj["bbox"]))
    
    return imageIds, BB

# Draw detection and annotation boxes in image
def drawBoxes(img, det, annot, colorDet, colorAnnot):
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


def randomFrame(videoPath):
    """
    This functions reads a video and returns a random frame and the number.

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