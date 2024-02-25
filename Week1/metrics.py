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


def mIoU(detection, anno, imagenames, classname):
    """
    This function computes the mean intersection over union of the detections and annotations

    Parameters
    ----------
    detection : tuple
        (image_ids, confs, BBoxes) format detections.
    anno : dict
        Dictionary with annotations.
    imagenames : list
        image names (frames).
    classname : str
        class name.

    Returns
    -------
    float
        mIoU value.

    """

    # first load gt
    # extract gt objects for this class
    class_recs = {}
    for imagename in imagenames:
        R = [obj for obj in anno[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        class_recs[imagename] = {"bbox": bbox}

    image_ids = detection[0]
    BB = detection[2]
    iou_images = np.zeros(len(imagenames))

    # For each image
    for i, imagename in enumerate(imagenames):

        # Get gt and det
        GTbbs = class_recs[imagename]["bbox"]
        indexDet = np.array(image_ids) == imagename
        bb = BB[indexDet, :].astype(float)

        det = [False] * bb.shape[0]

        iou_image = np.zeros(GTbbs.shape[0])
        # For each GT box
        for gt_bbox_i in range(GTbbs.shape[0]):
            BBGT = GTbbs[gt_bbox_i, :].astype(float)

            # Find best prediction
            max_iou = -1
            max_iou_ind = -1
            for pred_bbox_i in range(bb.shape[0]):

                predBB = bb[pred_bbox_i, :]

                iou_val = iou(BBGT, predBB)

                if iou_val > 0.01 and iou_val > max_iou and not det[pred_bbox_i]:
                    max_iou = iou_val
                    max_iou_ind = pred_bbox_i

            if max_iou_ind != -1:
                det[max_iou_ind] = True
                iou_image[gt_bbox_i] = max_iou

        # Add FP to iou_image
        iou_images[i] = np.sum(iou_image) * 2 / (iou_image.shape[0] + len(det))
        # iou_images[i] = np.mean(iou_image)

    return iou_images.mean()


def voc_eval(detection, anno, imagenames, classname, ovthresh=0.5, use_07_metric=True):
    """rec, prec, ap = voc_eval(detection,
                                anno,
                                imagenames,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: detections
        BBox detected
    annopath: annotations
        BBox annotations
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """

    # first load gt
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in anno[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        det = [False] * len(R)
        npos = npos + len(R)
        class_recs[imagename] = {"bbox": bbox, "det": det}

    image_ids = detection[0]
    confidence = detection[1]
    BB = detection[2]

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                    (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                    + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                    - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["det"][jmax]:
                tp[d] = 1.0
                R["det"][jmax] = 1
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def voc_ap(rec, prec, use_07_metric=True):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap