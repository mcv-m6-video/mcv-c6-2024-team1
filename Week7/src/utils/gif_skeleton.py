import cv2
import imageio
import os
import pickle 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# Read input skeleton Week7/frames/chew/AMADEUS_chew_h_nm_np1_fr_goo_7
skeletons_path = "data/hmdb51_2d.pkl"
with open(skeletons_path, "rb") as f:
    sk = pickle.load(f)



name = 'KELVIN_shoot_ball_u_cm_np1_ba_med_7'
folder_path = 'frames/shoot_ball/'
gif_writer = imageio.get_writer('plots/skeleton_'+name+'.gif', format='GIF', fps=20)
for ann in sk['annotations']:
    if ann['frame_dir'] == name:
        frames = sorted(os.listdir(folder_path + name +'/'))
        count = 0
        for frame in frames:
            try:
                img = plt.imread(folder_path+name+'/'+frame).astype(np.uint8)
                keypoints = ann['keypoint'][0,count,:]
                fig, ax = plt.subplots()
                ax.imshow(img, extent=[0, 320, 240, 0])
                ax.scatter(keypoints[:, 0], keypoints[:, 1], color='red', s=10)
                print(len(keypoints[:, 0]))
                canvas = FigureCanvas(fig)
        
                # Update the figure layout
                fig.tight_layout()
                # Draw the figure on the canvas
                canvas.draw()
                # Convert the canvas to a numpy array
                img_np = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
                img_np = img_np.reshape(canvas.get_width_height()[::-1] + (3,))
                gif_writer.append_data(img_np)
            except:
                print('jrjr')
                pass
            count+=1
            
gif_writer.close()       
        