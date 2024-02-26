import numpy as np


def iou(box1, box2):
    """
    Computes Intersection over Union (IoU) of two bounding boxes.
    Parameters:
        box1 (tuple): Coordinates of the first bounding box in the format (x1, y1, x2, y2).
        box2 (tuple): Coordinates of the second bounding box in the format (x1, y1, x2, y2).
    Returns:
        float: IoU of the two bounding boxes.
    """
    # Calculate the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # If the intersection rectangle is empty, return 0.0
    if x1 >= x2 or y1 >= y2:
        return 0.0
    
    # Calculate the area of intersection rectangle
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate the area of union of the two bounding boxes
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou

def mIoU(detection, gt):
    """
    Computes the mean Intersection over Union (mIoU) of a set of detections and ground truth.
    Parameters:
        detection (dict): Detections in the format {frame: [{'bbox': [...]}, ...]}.
        gt (dict): Ground truth in the format {frame: [{'name': ..., 'bbox': [...]}, ...]}.
    Returns:
        float: mIoU of the detections and ground truth.
    """
    # Initialize variables
    iou_images = np.array([])

    # For each frame
    for frame in gt.keys():
        # Get detections and ground truth
        if frame not in detection:
            iou_images = np.append(iou_images, np.zeros(len(gt[frame])))
            continue

        det = detection[frame]
        annot = gt[frame]

        # For each detection
        for det_obj in det:
            # Get detection box
            det_bbox = det_obj['bbox']
            max_iou = 0
            max_annot = None
            # For each annotation
            for annot_obj in annot:
                # Get annotation box
                annot_bbox = annot_obj['bbox']

                # Compute IoU
                iou_val = iou(det_bbox, annot_bbox)
                # We compute the maximum iou for each detection with the ground truth
                # and remove the detection with the highest iou from the list of ground truth
                if iou_val > max_iou:
                    max_iou = iou_val
                    max_annot = annot_obj
            if max_annot is not None:
                annot.remove(max_annot)

            iou_images = np.append(iou_images, max_iou)

        # If length of detections is different than length of ground truth
        if len(det) != len(annot):
            iou_images = np.append(iou_images, np.zeros(np.abs(len(annot) - len(det))))

    # Compute mIoU
    mIoU = np.mean(iou_images)

    return mIoU
