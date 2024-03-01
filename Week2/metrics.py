import numpy as np


def IoU(box1, box2):
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


def evaluate(detection, gt):
    """
    Computes some metrics given a set of detections and ground truth.
    Parameters:
        detection (dict): Detections in the format {frame: [{'bbox': [...]}, ...]}.
        gt (dict): Ground truth in the format {frame: [{'name': ..., 'bbox': [...]}, ...]}.
    Returns:
        float: mIoU of the detections and ground truth.
        float: Precision of the detections and ground truth.
        float: Recall of the detections and ground truth.
        float: F1 score of the detections and ground truth.
    """
    # Initialize variables
    iou_images = np.array([])
    tp = fp = fn = 0
    precision = []
    recall = []
    f1 = []

    # For each frame
    for frame in gt.keys():
        tp = 0  # True Positives
        fp = 0  # False Positives
        fn = 0  # False Negatives

        # Get detections and ground truth
        if frame not in detection:
            iou_images = np.append(iou_images, np.zeros(len(gt[frame])))
            # If there are no detections, all ground truth are False Negatives
            precision.append(0)
            recall.append(0)
            f1.append(0)
            continue

        det = detection[frame]
        annot = gt[frame].copy()

        # For each detection
        for det_obj in det:
            # Get detection box
            det_bbox = det_obj["bbox"]
            max_iou = 0
            max_annot_idx = -1

            # For each annotation
            for idx, annot_obj in enumerate(annot):
                # Get annotation box
                annot_bbox = annot_obj["bbox"]

                # Compute IoU
                iou_val = IoU(det_bbox, annot_bbox)

                # We compute the maximum iou for each detection with the ground truth
                # and remove the detection with the highest iou from the list of ground truth
                if iou_val > max_iou:
                    max_iou = iou_val
                    max_annot_idx = idx

            if max_annot_idx != -1:
                # Remove the annotation with the highest IoU from the list of ground truth
                annot.pop(max_annot_idx)

            iou_images = np.append(iou_images, max_iou)

            # Calculate True Positives, False Positives, and False Negatives
            if max_iou >= 0.5:  # Consider it a True Positive if IoU is greater than 0.5
                tp += 1
            else:
                fp += 1

        # Any remaining annotations are False Negatives
        fn += len(annot)

        # Compute precision, recall, and F1 score
        # We return directly a 0 if the denominator is 0,
        # it is, if there are no true positives, false positives or false negatives
        precision.append(tp / (tp + fp) if tp + fp > 0 else 0)
        recall.append(tp / (tp + fn) if tp + fn > 0 else 0)
        f1.append(
            2 * (precision[-1] * recall[-1]) / (precision[-1] + recall[-1])
            if precision[-1] + recall[-1] > 0
            else 0
        )

    # Compute the average IoU, precision, recall, and F1 score over all frames
    mIoU = np.mean(iou_images)
    precision = np.mean(precision)
    recall = np.mean(recall)
    f1_score = np.mean(f1)

    return mIoU, precision, recall, f1_score


def AP(annots_boxes, pred_boxes):
    # Initialize variables
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    gt_matched = np.zeros(len(annots_boxes))

    # Iterate over the predicted boxes
    for i, pred_box in enumerate(pred_boxes):
        ious = [IoU(pred_box, gt_box) for gt_box in annots_boxes]
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
    recall = tp / len(annots_boxes)
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


# Thanks group 5! :)
def mAP(annots, pred):
    """
    Compute the average precision (AP) of a model for a video
    :param annots: ground truth bounding boxes
    :param pred: predicted bounding boxes
    :return: list of average precision values
    """
    annots_list = []
    predictions_list = []
    for frame in annots.keys():
        annot_bboxes = annots[frame]
        pred_bboxes = pred.get(frame, [])
        predictions_list.append([bbox["bbox"] for bbox in pred_bboxes])
        annots_list.append([bbox["bbox"] for bbox in annot_bboxes])

    aps = []
    for _, (annots_boxes, pred_boxes) in enumerate(zip(annots_list, predictions_list)):
        aps.append(AP(annots_boxes, pred_boxes))

    return np.mean(aps)
