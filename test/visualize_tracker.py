import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.animation import FFMpegWriter
from skimage.draw import rectangle,rectangle_perimeter
from tqdm import trange
import matplotlib.transforms as mtransforms

from matplotlib.patches import FancyBboxPatch
import csv
from utils import *
def visualize_trackers(
    geoms,
    ids,
    videofile,
    colormap="jet",
    alpha=0.3,
    output_name="clip.mp4",
    engine="opencv",
):
    if engine not in ("opencv", "matplotlib"):
        raise ValueError(f"Unknown engine {engine}")

    ncolors = 10
    cmap = plt.cm.get_cmap(colormap, ncolors)
    reader = cv2.VideoCapture(videofile)

    def get_next_frame():
        _, img = reader.read()
        if img is None:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    shape = (
        int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )
    if engine == "opencv":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_name, fourcc, 25, shape[::-1])
        if writer.isOpened():
            for i in trange(len(geoms)):
                img = get_next_frame()
                
                if img is None or i not in geoms.keys():
                    continue
                for geom, id_ in zip(geoms[i], ids[i]):
                    rr, cc = rectangle_perimeter(( int(geom[1]), int(geom[0])), (int(geom[1]+geom[3]), int(geom[0]+geom[2])), shape=shape)
                    img[rr, cc] = cmap(int(id_) % ncolors, bytes=True)[:3]
                writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            writer.release()
        else:
            raise IOError("Oups")
    elif engine == "matplotlib":
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_frame_on(False)
        ax.axis("off")
        im = ax.imshow(np.empty(shape + (3,)))
        writer = FFMpegWriter(fps=25, codec="h264")
        with writer.saving(fig, output_name, dpi=100):
            for i in trange(len(geoms)):
                ax.patches = []
                ax.texts = []
                img = get_next_frame()
                if img is None or i not in geoms.keys():
                    continue
                im.set_array(img)
                for geom, id_ in zip(geoms[i], ids[i]):
                    color = cmap(id_ % ncolors, alpha=alpha)
                    bb = mtransforms.Bbox([[int(geom[0]),int(geom[1])], [int(geom[0])+int(geom[2]), int(geom[1])+int(geom[3])]])

                    p_bbox = FancyBboxPatch((bb.xmin, bb.ymin),
                                abs(bb.width), abs(bb.height),
                                boxstyle="square,pad=0.",
                                ec=color, fc='None', zorder=10.,color = color
                                )
                    ax.add_patch(p_bbox)
                    pos = int(geom[0]) ,int(geom[1])
                    ax.annotate(str(id_), pos,color='white')
                writer.grab_frame()
                
                
import cv2
import numpy as np
import os
from os.path import isfile, join
def save_video(video_name,frames):
    alltrack =[]
    with open('../results/'+video_name+'.txt' ) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            alltrack.append(row)
            
    alltrack = np.array(alltrack)
    alltrack = alltrack.astype('float64')
    geoms = alltrack[:,:6]
    ids = alltrack[:,:2]

    geoms = groupby(geoms,0)
    ids = groupby(ids,0)
    for k in geoms.keys():
        geoms[k] = geoms[k][:][:,2:6]
        ids[k] = ids[k][:][:,1]
    print('putting in video')
    
    print(video_name)
    
    frames_to_video(frames,'../videos/video'+video_name+'.mp4',fps=30)

    visualize_trackers(
        geoms,
        ids,
       '../videos/video'+video_name+'.mp4',
        colormap="jet",
        alpha=1,
        output_name="../videos/track"+video_name+".mp4",
        engine="matplotlib",
    )

def frames_to_video(frames,video_name,fps):
    img_array =[]
    for frame in frames:
        #reading each files
        img = cv2.imread(frame)
        height, width, layers = img.shape
        size = (width,height)
        #inserting the frames into an image array
        img_array.append(img)
    out = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(img_array)):
        # writing to a image array
        out.write(img_array[i])
    out.release()
