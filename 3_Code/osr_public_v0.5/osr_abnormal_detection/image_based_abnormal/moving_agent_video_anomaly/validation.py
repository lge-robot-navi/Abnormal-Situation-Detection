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



def validate_FC (FC_model, c3d_model, FC_model_criterion, opt, data_path, FC_model_batch_size, save_video_results, num_epoch):
    # FC_model.eval()

    x = glob.glob(data_path + '/*.mp4')
    input_file_list = open(r"validation", "w")

    for i in x:
        input_file_list.write(os.path.basename(i))
        input_file_list.write('\n')

    input_file_list.close()

    validation_files = []
    with open('./validation', 'r') as f:
        for row in f:
            validation_files.append(row[:-1])
    normal_files = 0
    anomaly_files = 0
    total_epoch_loss = []
    one_video_validation_loss = []
    print('Validating with validation videos: ')
    # print(len(validation_files))
    for num_of_file, validation_file in enumerate(validation_files):

        # print (num_of_file, end="")
        video_path = os.path.join(data_path, validation_file)
        video_level_annotation = -1
        if 'anomaly' in validation_file:
            # if 'anomaly' or 'assault' or 'burglary' in input_file:
            video_level_annotation = 1
            anomaly_files += 1
        # elif 'normal' or 'explosion' in input_file:
        elif 'normal' in validation_file:
            video_level_annotation = 0
            normal_files += 1
        else:
            raise ValueError('Validation Function: neither anomaly nor normal video string found in the video name')

        if os.path.exists(video_path):

            if os.path.exists('tmp_validation'):
                subprocess.call('rm -rf tmp_validation', shell=True)

            subprocess.call('mkdir tmp_validation', shell=True)
            subprocess.call('ffmpeg -i {} tmp_validation/image_%05d.jpg -hide_banner -nostats -loglevel panic'.format(video_path),
                            shell=True)
            class_names = []
            convolution_features_results = extract_features('tmp_validation', validation_file, class_names, c3d_model, opt,
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
                loss = FC_model_criterion(y_pred, y)

                validation_loss.append(loss.item())

                a = np.asarray(y_pred.cpu().detach().numpy().squeeze())
                FC_infered_labels.append(a)


                # FC_infered_labels.append(y_pred)
                # a = np.asarray(y_pred.cpu().detach().numpy().squeeze())
                # FC_infered_output.append(a)
            one_video_validation_loss.append(sum(validation_loss) / len(validation_loss))
            # FC_infered_output = np.concatenate(FC_infered_output, axis=0)
            #
            # FC_infered_labels = FC_infered_output >= 0.5
            FC_infered_labels = FC_infered_labels[0]


            if save_video_results is 1:
                labels_print_on_video(validation_file, opt.validation_video_root, './results/validation/epoch_' + str(num_epoch),
                                  FC_infered_labels, frame_interval=16)

        else:
            raise ValueError('validation video not found')
    total_epoch_validation_loss = sum(one_video_validation_loss) / len(one_video_validation_loss)
    return total_epoch_validation_loss



def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.data[0], inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch, i + 1, len(data_loader), batch_time=batch_time,
                  data_time=data_time, loss=losses, acc=accuracies))

    logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg
    })

    return losses.avg
