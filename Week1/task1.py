from utils import readXMLtoAnnotation, annoToDetecFormat, readTXTtoDet, randomFrame, drawBoxes
from metrics import voc_eval, mIoU
import numpy as np
from matplotlib import pyplot as plt

ANNOTATIONS_PATH = "../Data/ai_challenge_s03_c010-full_annotation.xml"
VIDEO_PATH = "../Data/AICity_data/train/S03/c010/vdo.avi"
DET_PATH = "../Data/AICity_data/train/S03/c010/det/"

def task1():
    className = "car"
    # Read GT annotations
    annot, imageNames = readXMLtoAnnotation(ANNOTATIONS_PATH)
        
    # Get noisy annotations
    # ...
    
    # Get imageIDs + BB
    imageIds, BB = annoToDetecFormat(annot, className)
    
    # Get random frame
    frame, frameNumber = randomFrame(VIDEO_PATH)
    
    # Plot noise
    colorDet = (255, 0, 0) # Red
    colorAnnot = (0, 0, 255) # Blue
    img_noise = drawBoxes(frame, BB[np.array(imageIds) == str(frameNumber),:], annot[str(frameNumber)], colorDet, colorAnnot, className)
    plt.imshow(img_noise)
    plt.show()
    
    # No confidence values, repeat N times with random values
    N = 10
    apSum = 0
    for i in range(N):
        conf = np.random.rand(len(imageIds))
        #print((imageIds, conf, BB))
        _,_, ap = voc_eval((imageIds, conf, BB), annot, imageNames, className)
        apSum += ap
    print("mAP: ", apSum/N)
    
    miou = mIoU((imageIds, conf, BB), annot, imageNames, className)
    print("mIoU: ", miou)
    
    # Task 1.2
    
    detMaskRcnn = "det_mask_rcnn.txt"
    detSSD = "det_ssd512.txt"
    detYolo3 = "det_yolo3.txt"
    
    imageIds, conf, BB = readTXTtoDet(DET_PATH + detMaskRcnn)
    _,_, ap = voc_eval((imageIds, conf, BB), annot, imageNames, "car")
    print("Mask RCNN mAP: ", ap)
    imageIds, conf, BB = readTXTtoDet(DET_PATH + detSSD)
    _,_, ap = voc_eval((imageIds, conf, BB), annot, imageNames, "car")
    print("SSD512 mAP: ", ap)
    imageIds, conf, BB = readTXTtoDet(DET_PATH + detYolo3)
    _,_, ap = voc_eval((imageIds, conf, BB), annot, imageNames, "car")
    print("Yolo3 mAP: ", ap)


if __name__ == '__main__':
    task1()