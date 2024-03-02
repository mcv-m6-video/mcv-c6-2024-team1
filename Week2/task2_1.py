import json

from week_utils import *

RESULTS_PATH = './results/'
FILE_IN = 'bbxs_clean.json'
FILE_OUT = 'bbxs_clean_tracked.json'


def cal_IoU(prev_tl, prev_br, new_tl, new_br):
    # Calculate coordinates of the intersection rectangle
    x_left = max(prev_tl[0], new_tl[0])
    y_top = max(prev_tl[1], new_tl[1])
    x_right = min(prev_br[0], new_br[0])
    y_bottom = min(prev_br[1], new_br[1])

    # If the intersection is valid (non-negative area), calculate the intersection area
    intersection_area = max(0, x_right - x_left + 1) * \
        max(0, y_bottom - y_top + 1)

    # Calculate areas of the individual bounding boxes
    prev_box_area = (prev_br[0] - prev_tl[0] + 1) * \
        (prev_br[1] - prev_tl[1] + 1)
    new_box_area = (new_br[0] - new_tl[0] + 1) * (new_br[1] - new_tl[1] + 1)

    # Calculate the union area
    union_area = prev_box_area + new_box_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


def track_max_overlap(file_in, file_out, iou_thr=0.4):
    """
    Object tracking by maximum overlap
    """
    with open(file_in, 'r') as f:
        data = json.load(f)

    bbxs = data
    f.close()

    # Iterate over each dictionary in the list
    for i in range(len(bbxs)):
        print(f'\nFrame {i}')

        track_ids = {}

        # Assign a different track to each object on the scene for first frame
        if i == 0:
            for k in bbxs[i]['xmin']:
                track_ids[k] = int(k)

        else:
            # Iterate over each bbx
            for k in bbxs[i]['xmin']:
                max_iou = -1
                max_idx = -1

                new_tl = (bbxs[i]['xmin'][k], bbxs[i]['ymin'][k])
                new_br = (bbxs[i]['xmax'][k], bbxs[i]['ymax'][k])

                # Iterate over each prev bbx
                for j in bbxs[i-1]['xmin']:
                    # Check if both bbxs are same class
                    if bbxs[i]['class'][k] == bbxs[i-1]['class'][j]:
                        prev_tl = (bbxs[i-1]['xmin'][j], bbxs[i-1]['ymin'][j])
                        prev_br = (bbxs[i-1]['xmax'][j], bbxs[i-1]['ymax'][j])
                        # print(f'Prev bbx: {prev_tl}, {prev_br}')
                        # print(f'New bbx: {new_tl}, {new_br}')

                        # Calculate IoU
                        iou = cal_IoU(prev_tl, prev_br, new_tl, new_br)
                        # print(f"IoU of track {j}: {iou}")

                        if iou > max_iou and iou > iou_thr:
                            max_iou = iou
                            max_idx = j
                            # print(f'max_iou {max_idx}: {max_iou}')

                # TODO(?): comprobar si el track ja s'ha posat a una altra bbx
                # Check if any bbx passed though the iou thershold
                if max_iou == -1:
                    # New track id
                    track_ids[k] = max(bbxs[i-1]['track'].values()) + 1
                    # print(f'New track: {track_ids[k]}')
                else:
                    # Put track id of bbx with max iou
                    track_ids[k] = bbxs[i-1]['track'][max_idx]
                    # print(f'Track followed: {track_ids[k]} with iou {max_iou}')

        bbxs[i]['track'] = track_ids
        # print(f'track_ids: {track_ids}')
        print(f"bbxs[i]['track']: {bbxs[i]['track']}")
        # print(f"bbxs[i]['track']: {bbxs[i]}")
        # if i == 3:
        #    break

    save_json(bbxs, file_out)

# TODO: visualization of tracking

if __name__ == "__main__":
    track_max_overlap(RESULTS_PATH + FILE_IN, RESULTS_PATH + FILE_OUT)
