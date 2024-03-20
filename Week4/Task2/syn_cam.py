import pickle
import numpy as np
import cv2

def load_pickle(pth: str):
    """Load a pickled tracklet file."""
    with open(pth, "rb") as f:
        res = pickle.load(f)
    return res

FOLDER_PATH = './Data/aic19-track1-mtmc-train/train/S03/'
OFFSETS = {'c010': 8.715,
'c011': 8.457,
'c012': 5.879,
'c013': 0,
'c014': 5.042,
'c015': 8.492}

def is_compatible(track1,track2, dtmin:int=-150,dtmax:int=150):
    cam1, cam2 = track1.cam, track2.cam
    # if there is no cam layout, we only check if the tracks are on the same camera
    # same camera -> they cannot be connected
    if cam1 == cam2:
        return False

    t1_start, t1_end = track1.global_start, track1.global_end
    t2_start, t2_end = track2.global_start, track2.global_end
    
    #TODO: REVISAR CONDICIONS
    if t2_start <= (t1_end + dtmax) and (t1_end + dtmin) <= t2_end:
        return True

    # check the track2 -> track1 transition too
    if t1_start <= (t2_end + dtmax) and (t2_end + dtmin) <= t1_end:
        return True
    
    if (t1_start<= t2_start and t2_start<= t1_end) or t2_start<= t1_start and t1_start<= t2_end:
        return True
    
    return False


CAMERAS = ['c010','c011','c012','c013','c014','c015']
class Camera():
    def __init__(self,file_path: str,  offset:float):
        print(file_path)
        self.video = cv2.VideoCapture(FOLDER_PATH+file_path+'/vdo.avi')
        self.tracks = load_pickle('./Week4/outputs/'+file_path+'/mot.pkl')
        #self.calibration
        self.offset = offset
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.start_frame = int( self.offset * self.fps )


def syncronize_trackers():

    cameras = [Camera(name, OFFSETS[name]) for name in CAMERAS]
    mini = min(OFFSETS.values())
    fps_minimum_OFFSET = next(cam.fps for cam in cameras if cam.offset == mini)

    #List of all tracks
    all_tracks =  []
    for i, cam_tracks in enumerate(cameras):
        for track in cam_tracks.tracks: 
            track.cam = i
            track.idx = len(all_tracks)
            track.global_start = int(( track.frames[0] / cam_tracks.fps + cam_tracks.offset)*fps_minimum_OFFSET)
            track.global_end = int((track.frames[-1] / cam_tracks.fps + cam_tracks.offset)*fps_minimum_OFFSET)
            all_tracks.append(track)
    n = len(all_tracks)

    #Compatibility
    compa = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            compa[i][j] = compa[j][i] = is_compatible(all_tracks[i], all_tracks[j])

    # Create dictionary to store idx of the trackers in each global frames
    global_frames = {}
    print(max(track.global_end for track in all_tracks))
    for frame in range(max(track.global_end for track in all_tracks)):
        global_frames[frame] = [[track.idx,track.cam] for track in all_tracks if track.global_start <= frame <= track.global_end]

    
    return all_tracks, compa, global_frames


