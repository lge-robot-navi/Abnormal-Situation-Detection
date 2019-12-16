#--input ./input --video_root ./videos/train --output none --model ./resnet-101-kinetics-ucf101_split1.pth --model_depth 101 --mode feature  #The parameters for this file
import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn
from pprint import pprint

from opts import parse_opts
from model import generate_model
from FC import FC
from mean import get_mean
# from classify import classify_video
from c3d_feature_extractor import extract_features
from validation import validate_FC
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
import glob
import cv2
import torch.utils.data as data_utils
from IO_video import labels_print_on_video


save_video_results = 1
save_video_interval_epochs = 3
FC_model_learning_rate = 0.05
# FC_model_decay_rate = 0.01
# FC_model_momentum = 0.9
FC_model_batch_size = 64
total_epochs = 300
resume_training = 0
load_FC_model = './FC_model/epoch_400'
FC_model_save_folder = './FC_model'
FC_save_model_epochs_interval = 10

learn_from_binary_annotations = 0

def check_accuracy(output, real_labels):

    output = output.cpu()

    temp_output = output.detach().numpy().astype(float)

    real_labels = real_labels.cpu()

    output_temp = output >= 0.5
    output_labels = output_temp.numpy().astype(int)
    y = real_labels.numpy()

    similar = np.isclose(output_labels, y).astype(int)

    return sum(similar) * 100 / real_labels.size(0)



def json_data_load(json_file_name):


    with open(json_file_name) as f:
        data = json.load(f)

    total_feature_vectors = 0
    # total_feature_vectors = 0
    for j in range(len(data)):

        total_feature_vectors = total_feature_vectors + len(data[j]["clips"])

        # for i in data[j]["clips"]:
        #     total_feature_vectors += 1



    np_data =  np.array([], dtype=np.float64).reshape(0, len(data[0]["clips"][0]["features"]))
    #np_y =  np.array([], dtype=np.float64).reshape(0, total_feature_vectors)
    np_y =  np.zeros((total_feature_vectors,1))
    count = 0
    for j in range(len(data)):

        for i in data[j]["clips"]:
            #print (i["scores"])
            a = np.asarray(i["features"])
            #print(a)
            np_data = np.vstack([np_data, a])

            y_temp = i["ground_truth_annotaion"]
            #y_temp = np.asarray(i["ground_truth_annotaion"])
            y_tempp = np.uint8(y_temp)
            np_y[count] = y_tempp
            count += 1
            print(count, ' / ', total_feature_vectors)


    print('end')

    #scaler = MinMaxScaler()

    #np_data = scaler.fit(np_data)
    np_data = minmax_scale(np_data, feature_range = (-1, 1), axis = 0)



    x = torch.from_numpy(np_data)
    y = torch.from_numpy(np_y)

    x = x.float()
    y = y.float()

    return x, y



if __name__=="__main__":

    # split = {'train', 'test'}
    split = {'train'}
    # torch.manual_seed(123)
    # torch.cuda.manual_seed(123)

    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 16
    opt.n_classes = 101 # 400

    all_epochs_train_loss_hist_for_final_graph = []
    all_epochs_validation_loss_hist_for_final_graph = []
    FC_model = FC().cuda()
    FC_model_criterion = torch.nn.MSELoss()
    # FC_model_optimizer = torch.optim.SGD(FC_model.parameters(), lr=FC_model_learning_rate,
    #                                      weight_decay=FC_model_decay_rate,
    #                                      momentum=FC_model_momentum)
    FC_model_optimizer = torch.optim.SGD(FC_model.parameters(), lr=FC_model_learning_rate)
    # FC_model_optimizer = torch.optim.Adam(FC_model.parameters(), lr=FC_model_learning_rate)

    previous_training_epoch = 0
    if resume_training is 1:   #resume training flag, if 1 then load existing model
        checkpoint = torch.load(load_FC_model)
        FC_model.load_state_dict(checkpoint['FC_model_state_dict'])
        FC_model_optimizer.load_state_dict(checkpoint['FC_optimizer_state_dict'])
        print('Loss of the loaded model is: ', checkpoint['loss'], ' and the training was stopped at Epoch ', checkpoint['epoch'])
        previous_training_epoch = checkpoint['epoch']


    x = glob.glob(opt.video_root + '/*.mp4') #path for the training videos.
    input_file_list = open(r"input", "w")

    for i in x:  #make a list of path and names of video files
        input_file_list.write(os.path.basename(i))
        input_file_list.write('\n')

    input_file_list.close()

    c3d_model = generate_model(opt) #C3D model for feature extraction
    # print('loading c3d_model {}'.format(opt.model))
    model_data = torch.load(opt.model)
    assert opt.arch == model_data['arch']
    # model.load_state_dict(model_data['state_dict'])
    c3d_model.load_state_dict(model_data['state_dict'], strict=True)
    c3d_model.eval()
    if opt.verbose:
        print(c3d_model)



    for num_epoch in range(total_epochs-previous_training_epoch):  #training loop
        num_epoch = num_epoch + previous_training_epoch

        if os.path.exists('tmp'):
            subprocess.call('rm -rf tmp', shell=True)


        input_files = []
        with open(opt.input, 'r') as f:
            for row in f:
                input_files.append(row[:-1])

        class_names = []
        with open('class_names_list') as f:
            for row in f:
                class_names.append(row[:-1])

        ffmpeg_loglevel = 'quiet'
        if opt.verbose:
            ffmpeg_loglevel = 'info'

        if os.path.exists('tmp'):
            subprocess.call('rm -rf tmp', shell=True)

        normal_files = 0
        anomaly_files = 0
        outputs = []
        one_epoch_loss = []
        for input_file in input_files:  #training videos loop

            video_path = os.path.join(opt.video_root, input_file)
            video_level_annotation = -1
            if 'anomaly' in input_file:
                video_level_annotation = 1
                anomaly_files += 1
            elif 'normal' in input_file:
                video_level_annotation = 0
                normal_files += 1
            else:
                raise ValueError('neither anomaly nor normal video string found in the video name')

            if os.path.exists(video_path):
                # print(video_path)
                subprocess.call('mkdir tmp', shell=True) #tmp directory is to save frames extracted from training video
                subprocess.call('ffmpeg -i {} tmp/image_%05d.jpg -hide_banner -nostats -loglevel panic'.format(video_path),
                                shell=True)

                convolution_features_results = extract_features('tmp', input_file, class_names, c3d_model, opt, video_level_annotation) #result is a features' ndarray of dimension: [total features, feature length]

                # --FCModel initialization and inference Code
                convolution_features_results_torch = torch.from_numpy(convolution_features_results)

                FC_inference_data = data_utils.TensorDataset(torch.from_numpy(convolution_features_results).cuda())
                FC_inference_loader = data_utils.DataLoader(
                    FC_inference_data,
                    FC_model_batch_size,
                    shuffle=False
                )

                # FC_infered_labels = []
                FC_infered_output = []
                first_FC_layer_outputs = []
                for batch_idx, x_batch in enumerate(FC_inference_loader):
                    # Forward pass: Compute predicted y by passing x to the model

                    # x_batch = x_batch.cuda()
                    # y_batch = y_batch.cuda()
                    x_batch = x_batch[0].float()
                    FC_model.eval()
                    y_pred, first_layer_output = FC_model(x_batch)
                    # FC_infered_labels.append(y_pred)
                    a = np.asarray(y_pred.cpu().detach().numpy().squeeze())
                    FC_infered_output.append(a)
                    first_FC_layer_outputs.append(first_layer_output)

                first_FC_layer_outputs = torch.cat(first_FC_layer_outputs)
                first_FC_layer_outputs = first_FC_layer_outputs.cpu().detach().numpy()
                FC_infered_output = np.concatenate(FC_infered_output, axis=0)

                FC_infered_labels = FC_infered_output >= 0.5
                FC_infered_labels = FC_infered_labels.astype(int)

                # --FCModel inference Code ends

                bp_labels = np.copy(FC_infered_labels)






                #FC Training part


                FC_training_data = data_utils.TensorDataset(torch.from_numpy(convolution_features_results),
                                                            torch.from_numpy(bp_labels))
                FC_train_loader = data_utils.DataLoader(
                    FC_training_data,
                    FC_model_batch_size,
                    shuffle=True
                )

                if learn_from_binary_annotations == 1:
                    if video_level_annotation == 1:
                        bp_labels = bp_labels * 0
                        bp_labels +=1

                    if video_level_annotation == 0:
                        bp_labels = bp_labels * 0


                #One loss per video
                #we already have inference from the network so we can now calculate error
                #we can create batch based on video size as well. for example if video is large then pick a fixed number of samples to check error

                train_loss = []
                if video_level_annotation is 1: #case of anomaly
                    for batch_idx, (x_batch, y) in enumerate(FC_train_loader):
                        # Forward pass: Compute predicted y by passing x to the model

                        x_batch = x_batch.cuda().float()
                        y = y.cuda().float()

                        if learn_from_binary_annotations == 1:
                            y = torch.tensor(np.ones(x_batch.size(0))).cuda().float()

                        FC_model.train()
                        y_pred, l1_output = FC_model(x_batch)
                        y_pred = y_pred.squeeze()


                        # Compute and print loss
                        loss = FC_model_criterion(y_pred, y)


                        # print(y_pred)
                        # Zero gradients, perform a backward pass, and update the weights.
                        FC_model_optimizer.zero_grad()

                        # perform a backward pass (backpropagation)
                        loss.backward()

                        # Update the parameters
                        FC_model_optimizer.step()

                        train_loss.append(loss.item())
                        # print('batch accuracy: ', check_accuracy(y_pred, y_batch))
                        # break
                    # torch.cuda.empty_cache()

                elif video_level_annotation is 0: #case if normal
                    for batch_idx, (x_batch, y) in enumerate(FC_train_loader):
                        # Forward pass: Compute predicted y by passing x to the model

                        x_batch = x_batch.cuda()
                        x_batch = x_batch.float()
                        y = torch.tensor(np.zeros(x_batch.size(0))).cuda().float()
                        FC_model.train()
                        y_pred, l1_output = FC_model(x_batch)
                        y_pred = y_pred.squeeze()

                        loss = FC_model_criterion(y_pred, y)
                        print('loss from network pred: ', (loss.item()))


                        # Zero gradients, perform a backward pass, and update the weights.
                        FC_model_optimizer.zero_grad()

                        # perform a backward pass (backpropagation)
                        loss.backward()

                        # Update the parameters
                        FC_model_optimizer.step()

                        train_loss.append(loss.item())
                        # print('batch accuracy: ', check_accuracy(y_pred, y_batch))
                    # torch.cuda.empty_cache()
                else:
                    raise ValueError('annotation is wrong')


                one_video_loss = sum(train_loss) / len(train_loss)
                print('Video: ', input_file, ' video level annotation: ', video_level_annotation, ' video loss: ', one_video_loss)

                one_epoch_loss.append(one_video_loss)

                # FC Training part ends

                # if num_epoch % save_video_interval_epochs is 0:
                #     if save_video_results is 1:
                #         result_to_print = np.vstack([, FC_infered_output])
                #         labels_print_on_video(input_file, opt.video_root, './results/train/epoch_' + str(num_epoch), labels=result_to_print,
                #                               frame_interval=16)  # labels should be for 16 (by default because of C3D) frames per one label


                subprocess.call('rm -rf tmp', shell=True)
            else:
                print('{} does not exist'.format(input_file))


        # print('total files: Anomalous = ', anomaly_files, 'Normal = ', normal_files)
        if os.path.exists('tmp'):
            subprocess.call('rm -rf tmp', shell=True)

        ##################################################################################################################write code for validation loss
        ##################################################################################################################write code for writing video with output results

        avg_epoch_loss = sum(one_epoch_loss) / len(one_epoch_loss)
        print('Epoch [{}/{}], Training Loss:{:.4f}'.format(num_epoch + 1, total_epochs,
                                                                                         avg_epoch_loss))
        if num_epoch % save_video_interval_epochs is not 0:
            save_video_results = 0
        one_epoch_validation_loss = validate_FC(FC_model=FC_model, c3d_model=c3d_model,
                                                FC_model_criterion=FC_model_criterion, opt=opt,
                                                data_path='./videos/validation',
                                                FC_model_batch_size=FC_model_batch_size, save_video_results = save_video_results, num_epoch=num_epoch)
        save_video_results = 1

        print ('Validation Loss : ', one_epoch_validation_loss)
        all_epochs_train_loss_hist_for_final_graph.append(avg_epoch_loss)
        all_epochs_validation_loss_hist_for_final_graph.append(one_epoch_validation_loss)
        ## Code for run time plot update
        ##uncomment this for runtime plotting
        plt.title("Training Error Graph")
        plt.xlabel("Training Epochs")
        plt.ylabel("Training Error")
        plt.scatter(num_epoch + 1, avg_epoch_loss, linewidths=0.5, c='red', marker='.')
        plt.scatter(num_epoch + 1, one_epoch_validation_loss,  linewidths=0.5, c='blue', marker='.')
        plt.pause(0.00001)

        if num_epoch % FC_save_model_epochs_interval is 0:
            print('saving model...')
            model_save_path = FC_model_save_folder + '/epoch_' + str(num_epoch + 1)
            torch.save({
                'epoch': num_epoch+1,
                'FC_model_state_dict': FC_model.state_dict(),
                'FC_optimizer_state_dict': FC_model_optimizer.state_dict(),
                'loss': all_epochs_train_loss_hist_for_final_graph,
            }, model_save_path)


    plt.clf()
    shist = [h for h in all_epochs_train_loss_hist_for_final_graph]
    vhist = [h for h in all_epochs_validation_loss_hist_for_final_graph]
    plt.title('training_loss')
    plt.xlabel("Training Epochs")
    plt.ylabel("Error")
    plt.plot(range(1, total_epochs + 1), shist, label="train")
    plt.plot(range(1, total_epochs + 1), vhist, label="validation")
    # plt.ylim((0,1.))
    # plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    plt.savefig('training_loss' + '.png')

    plt.show()
    plt.clf()
    print('end')

    print('end')