import moviepy.editor as mp
from moviepy.video.fx.all import crop


import glob
import os.path




#videos_path = '/media/zaigham/Data4/normal_no_human_videos'

# x = glob.glob("/home/zaigham/Desktop/FC test videos/**/*.mp4")
x = glob.glob("/media/zaigham/SSD_1TB/Pohang dataset/normal/*.mp4")
target_dir = '/media/zaigham/SSD_1TB/Pohang dataset/center_cropped/normal/'


#Use os.path.basename(path) to get the filename.




num = 1

for i in x:

    clip = mp.VideoFileClip(i)
    (w, h) = clip.size
    cropped_clip = crop(clip, width=h, height=h, x_center=w / 2, y_center=h / 2)
    # cropped_clip.write_videofile(target_dir + 'anomaly' + str(num)+'.mp4')


    clip_resized = cropped_clip.resize((112,112)) # make the height 360px ( According to moviePy documenation The width is then computed so that the width/height ratio is conserved.)
    clip_resized.write_videofile(target_dir + 'anomaly' + str(num)+'.mp4')
    num += 1
