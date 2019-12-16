import torch
from torch.autograd import Variable
import time
import sys
import os
import subprocess
from c3d_feature_extractor import extract_features
import numpy as np
import glob
import torch.utils.data as data_utils
from IO_video import labels_print_on_video
# from utils import AverageMeter, calculate_accuracy


def max_anomaly_scores_in_video (FC_model, c3d_model, FC_model_criterion, opt, video_name, data_path, FC_model_batch_size, save_video_results, num_epoch):
    # FC_model.eval()

    # x = glob.glob(data_path + '/*.mp4')
    # input_file_list = open(r"validation", "w")
    #
    # for i in x:
    #     input_file_list.write(os.path.basename(i))
    #     input_file_list.write('\n')
    #
    # input_file_list.close()
    #
    # validation_files = []
    # with open('./validation', 'r') as f:
    #     for row in f:
    #         validation_files.append(row[:-1])
    normal_files = 0
    anomaly_files = 0
    # total_epoch_loss = []
    # one_video_validation_loss = []
    # print('Validating with validation videos: ')
    # # print(len(validation_files))
    # for num_of_file, validation_file in enumerate(validation_files):

    # print (num_of_file, end="")
    video_path = os.path.join(data_path, video_name)
    video_level_annotation = -1
    if 'anomaly' in video_name:
        # if 'anomaly' or 'assault' or 'burglary' in input_file:
        video_level_annotation = 1
        anomaly_files += 1
    # elif 'normal' or 'explosion' in input_file:
    elif 'normal' in video_name:
        video_level_annotation = 0
        normal_files += 1
    else:
        raise ValueError('Validation Function: neither anomaly nor normal video string found in the video name')
    max_anomaly_score = 0
    if os.path.exists(video_path):

        if os.path.exists('tmp_validation'):
            subprocess.call('rm -rf tmp_validation', shell=True)

        subprocess.call('mkdir tmp_validation', shell=True)
        subprocess.call('ffmpeg -i {} tmp_validation/image_%05d.jpg -hide_banner -nostats -loglevel panic'.format(video_path),
                        shell=True)
        class_names = []
        convolution_features_results = extract_features('tmp_validation', video_name, class_names, c3d_model, opt,
                                                        video_level_annotation)  # result is an ndarray of dimension: [total features, feature length]


        FC_inference_validation_data = data_utils.TensorDataset(torch.from_numpy(convolution_features_results).cuda())
        FC_inference_validation_data_loader = data_utils.DataLoader(
            FC_inference_validation_data,
            FC_model_batch_size,
            shuffle=False
        )



        FC_infered_labels = []
        validation_loss = []
        for batch_idx, x_batch in enumerate(FC_inference_validation_data_loader):
            # Forward pass: Compute predicted y by passing x to the model
            y = torch.tensor(np.zeros(x_batch[0].size(0))).cuda().float()

            if video_level_annotation is 1:
                y = torch.tensor(np.ones(x_batch[0].size(0))).cuda().float()
            # x_batch = x_batch.cuda()
            # y_batch = y_batch.cuda()
            x_batch = x_batch[0].float()
            FC_model.eval()
            y_pred, garbage = FC_model(x_batch)
            y_pred = y_pred.squeeze()

            if y_pred.dim()>0:
                if max(y_pred) > max_anomaly_score:
                    max_anomaly_score   = max(y_pred)
            loss = FC_model_criterion(y_pred, y)

            validation_loss.append(loss.item())

            a = np.asarray(y_pred.cpu().detach().numpy().squeeze())
            FC_infered_labels.append(a)


            # FC_infered_labels.append(y_pred)
            # a = np.asarray(y_pred.cpu().detach().numpy().squeeze())
            # FC_infered_output.append(a)
        # one_video_validation_loss.append(sum(validation_loss) / len(validation_loss))
        # FC_infered_output = np.concatenate(FC_infered_output, axis=0)
        #
        # FC_infered_labels = FC_infered_output >= 0.5
        FC_infered_labels = FC_infered_labels[0]


        # if save_video_results is 1:
        #     labels_print_on_video(validation_file, opt.validation_video_root, './results/validation/epoch_' + str(num_epoch),
        #                       FC_infered_labels, frame_interval=16)

    else:
        raise ValueError('validation video not found')
    # total_epoch_validation_loss = sum(one_video_validation_loss) / len(one_video_validation_loss)
    return [max_anomaly_score, video_level_annotation]
