import cv2

# Extract all frame from the video


vidcap = cv2.VideoCapture('ab_ped_01.mp4')
success, image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frame_extract/ab_ped_01/frame%d.jpg" % count, image)     # save frame as JPEG file
  success, image = vidcap.read()
  count += 1
print("Done.")