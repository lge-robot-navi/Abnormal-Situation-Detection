#!/usr/bin/env python

#converts videos to imahes

import cv2
import os
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize, VideoCapture, imwrite



print(cv2.__version__)

# out_dir = '/media/zaigham/SSD_1TB/etri_db/w_humans/adaptive_Histogram_equilized_frames_10fps'
# source_dir = '/media/zaigham/SSD_1TB/etri_db/w_humans/vids'
# filenames = ['combined_anomaly_with_gimbal_1_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_anomaly_with_gimbal_3_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_anomaly_with_gimbal_4_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_anomaly_with_gimbal_6_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_anomaly_with_gimbal_7_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_anomaly_with_gimbal_8_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_anomaly_with_gimbal_9_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_anomaly_with_gimbal_11_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_anomaly_with_gimbal_12_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_anomaly_with_gimbal_13_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_anomaly_with_gimbal_15_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_anomaly_with_gimbal_16_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_anomaly_with_gimbal_18_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_anomaly_without_gimbal_2_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_anomaly_without_gimbal_3_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_anomaly_without_gimbal_4_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_anomaly_without_gimbal_6_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_anomaly_without_gimbal_7_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_anomaly_without_gimbal_8_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_anomaly_without_gimbal_9_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_anomaly_without_gimbal_11_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_anomaly_without_gimbal_12_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_anomaly_without_gimbal_14_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_anomaly_without_gimbal_15_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_normal_with_gimbal_2_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_normal_with_gimbal_4_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_normal_with_gimbal_5_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_normal_with_gimbal_7_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_normal_with_gimbal_8_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_normal_with_gimbal_9_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_normal_with_gimbal_10_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_normal_without_gimbal_1_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_normal_without_gimbal_3_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_normal_without_gimbal_4_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_normal_without_gimbal_6_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_normal_without_gimbal_7_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_normal_without_gimbal_8_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_normal_without_gimbal_9_epochs__0_10_20_25_35_50_100_149.avi',
#              'combined_normal_without_gimbal_11_epochs__0_10_20_25_35_50_100_149.avi',
#              ]

filenames = ['combined_anomaly2_epochs__0_10_20_25_50_100_149.avi',
             'combined_anomaly3_epochs__0_10_20_25_50_100_149.avi',
             'combined_anomaly5_epochs__0_10_20_25_50_100_149.avi',
             'combined_anomaly6_epochs__0_10_20_25_50_100_149.avi',
             'combined_anomaly7_epochs__0_10_20_25_50_100_149.avi',
             'combined_anomaly8_epochs__0_10_20_25_50_100_149.avi',
             'combined_anomaly9_epochs__0_10_20_25_50_100_149.avi',
             'combined_anomaly10_epochs__0_10_20_25_50_100_149.avi',
             'combined_anomaly11_epochs__0_10_20_25_50_100_149.avi',
             'combined_anomaly12_epochs__0_10_20_25_50_100_149.avi',
             'combined_anomaly13_epochs__0_10_20_25_50_100_149.avi',
             'combined_anomaly14_epochs__0_10_20_25_50_100_149.avi',
             'combined_normal1_epochs__0_10_20_25_50_100_149.avi',
             'combined_normal3_epochs__0_10_20_25_50_100_149.avi',
             'combined_normal6_epochs__0_10_20_25_50_100_149.avi',
             'combined_normal9_epochs__0_10_20_25_50_100_149.avi',
             'combined_normal10_epochs__0_10_20_25_50_100_149.avi',
             'combined_normal11_epochs__0_10_20_25_50_100_149.avi',
]


for filename in filenames:

        video_paths = [#'/home/zaigham/Desktop/Anomaly detetction/videos/ETRI_outdoor_2_origional/anomaly/2019_09_05 21_24(1).mp4',
                        '/home/zaigham/Desktop/Anomaly detetction/results_iteration in temporal order_zero_temporal and instance factors/',
                        '/home/zaigham/Desktop/Anomaly detetction/results_iteration in temporal order_point_000008_temporal and instance factors/']
        # videos_names = ['anomaly1','anomaly4','anomaly15','normal2','normal5','normal7']

        text = ['Gimbal_dataset_only_no_temporal_instance_loss',
                'Gimbal_dataset_only_0.000008_temporal_instance_loss'
        ]


        # video_name = 'anomaly2'
        # video_extension = '.mp4'
        # epochs = [0,10,20,25,50,100,149]
        total_number_of_videos = len(video_paths)

        # ep = str(epochs[0])

        # videos_path = '/home/zaigham/Desktop/Anomaly detetction/results_iteration in temporal order_point_000008_temporal and instance factors/train/'
        output_path = '/home/zaigham/Desktop/Anomaly detetction/results_iteration in temporal order_point_000008_temporal and instance factors/'
        vid = [None for x in range(total_number_of_videos)]
        for i,j in enumerate(video_paths):
                 vid[i] = video_paths[i] + filename

        # histogram_eq = 0
        # adaptive_hist_eq = 1
        # videos = ['20180713_101224.mp4', '']

        # file_list = os.listdir(source_dir)
        # count = 0
        dilation_track_var = 0

        vidcap = [None for x in range(total_number_of_videos)]
        image = [None for x in range(total_number_of_videos)]

        for i,j in enumerate(vid):
                vidcap[i] = cv2.VideoCapture(vid[i])
                success, image[i] = vidcap[i].read()
                cv2.putText(image[i], text[i], (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),1)
                if not success:
                        raise Exception('video not found')

        format = "XVID"
        fourcc = VideoWriter_fourcc(*format)

        # fourcc = cv2.VideoWriter_fourcc('mp4')  # Be sure to use lower case
        # out = cv2.VideoWriter('stitched.mp4', fourcc, 30, (1920*2, 1080))
        # out_path = '/media/zaigham/SSD_1TB/gits/context_encoder_pytorch/Video_test/Codes and Trained Network/patches/final/'
        size = image[0].shape[0] * total_number_of_videos,image[0].shape[1]
        size = image[0].shape[1],image[0].shape[0] * total_number_of_videos

        output_file_name = 'vertical_combined'+filename
        # for i,j in enumerate(epochs):
        #         output_file_name= output_file_name + '_' + str(epochs[i])

        output_path_and_file_name = output_path + output_file_name


        vid_writer = VideoWriter(output_path_and_file_name, fourcc, float(30), size, True)

        while success:

                #for ch in range(3):
                #image2[:,:,ch] = cv2.equalizeHist(image[:, :, ch])

                final_image = np.concatenate((image[0], image[1]), axis=0)
                for i in range(len(image)-2):
                        final_image = np.concatenate((final_image, image[i+2]), axis=1)




                #image_path = os.path.join(out_path, '/output')
                #frame = cv2.imread('')
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                vid_writer.write(final_image)  # Write out frame to video

                # cv2.imshow('video', image_1)
                # imwrite ('/media/zaigham/SSD_1TB/gits/context_encoder_pytorch/Video_test/Codes and Trained Network/patches/final/1.jpg', image)

                for i in range(total_number_of_videos):
                        success, image[i] = vidcap[i].read()
                        cv2.putText(image[i], text[i], (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),1)

                print('Read a new frame: ', dilation_track_var, success)
                dilation_track_var += 1


        print ('job finished')


        # Release everything if job is finished
        vid_writer.release()
