import cv2

# Extract all frame from the video
vid_name = 'ab_ped_01.mp4'
vidcap = cv2.VideoCapture(vid_name)

out_path = 'frame_extract'
success, image = vidcap.read()

count = 0
out_file = out_path + vid_name.replace('.mp4','')
while success:
  cv2.imwrite(out_file + "frame%d.jpg" % count, image)     # save frame as JPEG file
  success, image = vidcap.read()
  count += 1
