import imageio
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List
import argparse
import pickle

from inverse_projection import MultiCameraTracklet, Tracklet


STATIC_ATTRIBUTES = {
    "color": ["yellow", "orange", "green", "gray", "red", "blue", "white", "golden", "brown", "black",
              "purple", "pink"],
    "type": ["sedan", "suv", "van", "hatchback", "mpv",
             "pickup", "bus", "truck", "estate", "sportscar", "RV", "bike"],
}

DYNAMIC_ATTRIBUTES = {
    "brake_signal": ["off", "on"],
}

def get_attribute_value(name: str, value: int):
    """Get the description of an attribute, e.g. get_attribute_value('color', 5) -> 'blue'."""
    if name == "speed":
        return str(value)
    if name in STATIC_ATTRIBUTES:
        return STATIC_ATTRIBUTES[name][value]
    if name in DYNAMIC_ATTRIBUTES:
        return DYNAMIC_ATTRIBUTES[name][value]
    err = f"Invalid static or dynamic attribute name: {name}."
    raise ValueError(err)

def put_text(img_pil, text, x, y, color, font):
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y), text, (color[0], color[1], color[2],
                             255), font=font)
    return img_pil

def annotate(img_pil, id_label, attributes, tx, ty, bx, by, color, font):
    """ Put the id label and the features as text below or above of a bounding box. """

    draw = ImageDraw.Draw(img_pil, "RGBA")
    draw.rectangle([tx, ty, bx, by], outline=color, width=3)
    text = [id_label] + [f"{k}: {get_attribute_value(k, v)}" for k,
                         v in attributes.items()]
    text = "\n".join(text)

    textcoords = draw.multiline_textbbox((tx, by), text, font=font)

    # if the annotation below the box stretches out of the image, put it above
    if textcoords[3] >= img_pil.size[1]:
        txt_y = ty - (textcoords[3] - textcoords[1]) - 4
    else:
        txt_y = by

    # draw rectangle in the background
    coords = draw.multiline_textbbox((tx, txt_y), text, font=font)
    # add some padding
    textcoords = (coords[0] - 2, coords[1] - 2, coords[2] + 2, coords[3] + 2)
    draw.rectangle(textcoords, fill=color)

    # draw the text finally
    draw.multiline_text((tx, txt_y), text, (0, 0, 0), font=font)
    return img_pil


class Video:
    def __init__(self, font, fontsize=13):
        cmap = plt.get_cmap("Set2")
        self.colors = [cmap(i)[:3] for i in range(cmap.N)]
        cmap2 = plt.get_cmap("hsv")
        for i in np.linspace(0.1, 0.5, 7):
            self.colors.append(cmap2(i)[:3])
        self.HASH_Q = int(1e9 + 7)

        try:
            self.font = ImageFont.truetype(font, fontsize)
        except OSError:
            #log.error(f"Video: Font {font} cannot be loaded, using PIL default font.")
            print(f"Video: Font {font} cannot be loaded, using PIL default font.")
            self.font = ImageFont.load_default()
        self.frame_font = ImageFont.truetype(font, 18)
        self.frame_num = 0

    def render_tracks(self, frame, track_ids, track_bboxes, attributes):
        overlay = Image.fromarray(
            np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8), "RGBA")
        for track_id, bbox, attrib in zip(track_ids, track_bboxes, attributes):
            tx, ty, w, h = bbox
            bx, by = int(tx + w), int(ty + h)
            color = self.colors[(self.HASH_Q * int(track_id)) % len(self.colors)]
            color = tuple(int(i * 255) for i in color)

            overlay = annotate(overlay, str(track_id), attrib,
                               tx, ty, bx, by, color, self.font)

        mask = Image.fromarray((np.array(overlay) > 0).astype(np.uint8) * 192)
        frame_img = Image.fromarray(frame)
        frame_img.paste(overlay, mask=mask)

        put_text(frame_img, f"Frame {self.frame_num}",
                 0, 0, (255, 0, 0), self.frame_font)
        self.frame_num += 1

        return np.array(frame_img)

class FileVideo(Video):
    def __init__(self, font, save_path, fps, codec, format="FFMPEG", mode="I", fontsize=13):
        super().__init__(font, fontsize=fontsize)
        self.video = imageio.get_writer(save_path, format=format, mode=mode,
                                        fps=fps, codec=codec, macro_block_size=8)

    def update(self, frame, track_ids, bboxes, attributes):
        frame = self.render_tracks(frame, track_ids, bboxes, attributes)
        self.video.append_data(frame)

    def close(self):
        self.video.close()


def get_tracks_by_cams(multicam_tracks: List[MultiCameraTracklet]) -> List[List[Tracklet]]:
    """Return multicam tracklets sorted by cameras."""
    if len(multicam_tracks) == 0:
        return []
    tracks_per_cam = [[] for _ in range(multicam_tracks[0].n_cams)]
    for mtrack in multicam_tracks:
        for track in mtrack.tracks:
            tracks_per_cam[track.cam].append(track)
    return tracks_per_cam

def annotate_video_with_tracklets(input_path, output_path, tracklets, font="Hack-Regular.ttf",
                                  fontsize=13):
    video_in = imageio.get_reader(input_path)
    video_meta = video_in.get_meta_data()
    video_out = FileVideo(
        font, output_path, video_meta["fps"], video_meta["codec"], fontsize=fontsize)

    tracklets = sorted(tracklets, key=lambda tr: tr.frames[0])
    active_tracks = {}
    nxt_track = 0

    for frame_idx, frame in enumerate(video_in):
        while nxt_track < len(tracklets) and tracklets[nxt_track].frames[0] == frame_idx:
            active_tracks[nxt_track] = 0
            nxt_track += 1

        track_ids, bboxes, attribs = [], [], []
        ended_tracks = []
        incr_tracks = []

        # gather info for the current frame
        for track_idx, ptr in active_tracks.items():
            track = tracklets[track_idx]

            try:
                static_refined = isinstance(
                    next(iter(track.static_attributes.values())), int)
            except StopIteration:
                static_refined = True

            if track.frames[ptr] == frame_idx:
                track_ids.append(track.track_id)
                bboxes.append(track.bboxes[ptr])

                attr = {}
                for k, v in track.static_attributes.items():
                    if static_refined:
                        attr[k] = v
                    else:
                        attr[k] = v[ptr]
                for k, v in track.dynamic_attributes.items():
                    attr[k] = v[ptr]
                attribs.append(attr)

                if ptr >= len(track.frames) - 1:
                    ended_tracks.append(track_idx)
                else:
                    incr_tracks.append(track_idx)

        for track_idx in ended_tracks:
            del active_tracks[track_idx]
        for track_idx in incr_tracks:
            active_tracks[track_idx] += 1

        video_out.update(frame, track_ids, bboxes, attribs)

    video_out.close()

def annotate_video_mtmc(video_in, video_out, multicam_tracks, cam_idx, **kwargs):
    tracks = get_tracks_by_cams(multicam_tracks)[cam_idx]
    annotate_video_with_tracklets(video_in, video_out, tracks, **kwargs)


def load_mtmc_tracklets(path: str):
    with open(path, "rb") as f:
        res = pickle.load(f)
    return res

def save_mtmc_tracklets(multicam_tracks: List[MultiCameraTracklet], path: str):
    with open(path, "wb") as f:
        pickle.dump(multicam_tracks, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="annotate a video from multicam tracks")
    parser.add_argument("--cam_idx", type=int, required=True)
    parser.add_argument("--tracklets", required=True,
                        help="multicam tracklets pickle")
    parser.add_argument("--video_in", required=True)
    parser.add_argument("--video_out", required=True)
    args = parser.parse_args()

    tracks = load_mtmc_tracklets(args.tracklets)
    annotate_video_mtmc(args.video_in, args.video_out, tracks, args.cam_idx)