import glob
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
import osr_functions as ofn


global map_x_size, map_y_size, map_layer_num
map_x_size = 170
map_y_size = 140
map_layer_num = 12

ofn.prob_hist_initialize()
ofn.ae_obj_initialize()
ofn.cnn_obj_initialize()
ofn.motion_prediction_initialize()
status_list = ['data-time', 'hum_num', 'hum_spd', 'car_num', 'car_spd', 'prob_min', 'pos_y', 'pos_x', 'ae_error', 'cnn_n', 'cnn_a', 'motion_err']

map_files = glob.glob('mapDB_v2/0/npy/*.npy')
data_length = np.shape(map_files)[0]
map_obs_prb = 1 - plt.imread("output/obs_prob_map.png")[:, :, 0]  # object probability report map
obs_pos = np.argwhere(map_obs_prb > 0.1)
obs_pos_len = np.shape(obs_pos)[0]

idx = 0
while idx < 5000:
    idx_rand = int(0.5*data_length * np.random.rand(1))+int(0.5*data_length)
    map_data = np.nan_to_num(np.load(map_files[idx_rand]))
    # map_data[np.isnan(map_data)] = 0
    map_data[:, :, 11] = 1

    hum_num_syn = int(5 * np.random.rand(1) + 1)
    car_num_syn = int(2 * np.random.rand(1) + 0)

    # Abnormal @ fixed_1~3
    # agn_center = [[90, 105], [85, 45], [35, 35]]
    # pos_agn = agn_center[np.clip(int(2.99*np.random.rand(1)), 0, 2)]
    # pos_y = 140 - pos_agn[0]
    # pos_x = pos_agn[1]

    # Abnormal @ entire area
    # pos_y = np.clip(int(map_y_size * np.random.rand(1)), 0, map_y_size)
    # pos_x = np.clip(int(map_x_size * np.random.rand(1)), 0, map_x_size)

    # Abnormal @ moving agent patrol area
    pos_y = obs_pos[int(obs_pos_len * np.random.rand(1))][0]
    pos_x = obs_pos[int(obs_pos_len * np.random.rand(1))][1]

    # Initial position of human and car
    hum_pos_y = np.zeros((hum_num_syn))
    hum_pos_x = np.zeros((hum_num_syn))
    car_pos_y = np.zeros((car_num_syn))
    car_pos_x = np.zeros((car_num_syn))
    for cnt in range (hum_num_syn):
        hum_pos_y[cnt] = int(pos_y + 6 * (np.random.rand(1)-0.5))
        hum_pos_x[cnt] = int(pos_x + 6 * (np.random.rand(1)-0.5))
    for cnt in range (car_num_syn):
        car_pos_y[cnt] = int(pos_y + 6 * (np.random.rand(1)-0.5))
        car_pos_x[cnt] = int(pos_x + 6 * (np.random.rand(1)-0.5))

    # # Random velocity and change position of human and car
    map_stack_img = np.zeros((35*6, 84, 3))
    for frame in range(6):
        map_data = np.load(map_files[idx_rand])
        map_data[:, :, 11] = 1
        for idx_h in range(hum_num_syn):
            vel_y = float(5 * (np.random.rand(1)-0.5))
            vel_x = float(5 * (np.random.rand(1)-0.5))
            dist_y = 1.0 * vel_y
            dist_x = 1.0 * vel_x
            hum_pos_y[idx_h] = int(np.clip(hum_pos_y[idx_h] + dist_y, 0, map_y_size-1))
            hum_pos_x[idx_h] = int(np.clip(hum_pos_x[idx_h] + dist_x, 0, map_x_size-1))
            map_data[hum_pos_y[idx_h], hum_pos_x[idx_h], 0] = 1.0
            map_data[hum_pos_y[idx_h], hum_pos_x[idx_h], 1] = vel_x
            map_data[hum_pos_y[idx_h], hum_pos_x[idx_h], 2] = vel_y
        for idx_c in range(car_num_syn):
            vel_y = float(5 * (np.random.rand(1)-0.5))
            vel_x = float(5 * (np.random.rand(1)-0.5))
            dist_y = 1.0 * vel_y
            dist_x = 1.0 * vel_x
            car_pos_y[idx_c] = int(np.clip(car_pos_y[idx_c] + dist_y, 0, map_y_size-1))
            car_pos_x[idx_c] = int(np.clip(car_pos_x[idx_c] + dist_x, 0, map_x_size-1))
            map_data[car_pos_y[idx_c], car_pos_x[idx_c], 4] = 1.0
            map_data[car_pos_y[idx_c], car_pos_x[idx_c], 5] = vel_x
            map_data[car_pos_y[idx_c], car_pos_x[idx_c], 6] = vel_y

        # Observation and probability map generation
        map_obj, hum_num, hum_spd_max, car_num, car_spd_max, prob_min, prob_min_pos, agent_id, obj_type = ofn.map_obj_generation(map_data)
        map_obj_reduced = ofn.prob_map_reducer(ofn.prob_map_reducer(map_obj))
        maha_nor, maha_abn = ofn.cnn_maha_dist_obj(np.dstack((map_obj_reduced[:, 0:42, :], map_obj_reduced[:, 42:84, :])))
        obj_ae_err = ofn.ae_obj_recon(map_obj_reduced[:, 0:42, :])
        map_stack_img[frame*35:(frame+1)*35, :, :] = map_obj_reduced

        motion_err = ofn.motion_prediction_error(map_stack_img.astype(np.uint8)[:, :, ::-1])
        file_name_status = '_H'+str(hum_num)+'_'+str(hum_spd_max)+'C'+str(car_num)+'_'+str(car_spd_max)+'P'+str(prob_min)+str(prob_min_pos)+'AE'+str(obj_ae_err)+'CNN'+str(maha_nor)+'_'+str(maha_abn) + 'MP'+str(motion_err)
        aty_criteria = [prob_min, obj_ae_err, maha_nor, maha_abn, motion_err]
        print(idx, frame, hum_num, car_num, aty_criteria)

#     # Save abnormal data
    if prob_min < 2.5 and obj_ae_err > 3.0:
#     if prob_min < 3.0 and maha_nor > maha_abn and obj_ae_err > 3.0:
    # if prob_min < 3.0 and obj_ae_err > 3.0 and motion_err > 3.7:
        print(idx, file_name_status)
        idx += 1
        filename_img = 'mapDB_v2/1/obj/abn_map_obj_' + str(idx) + '_' + str(0) + file_name_status + '.bmp'
        cv2.imwrite(filename_img, map_stack_img[0:35, :, :].astype(np.uint8)[:, :, ::-1])
        # filename_stack_img = 'mapDB_v2/1/mtn/abn_map_mtn_' + str(idx) + '_' + str(frame) + file_name_status + '.bmp'
        # cv2.imwrite(filename_stack_img, map_stack_img.astype(np.uint8)[:, :, ::-1])

        # filename_npy = 'mapDB/Abnormal/1npy/abn_map_obj_' + str(idx) + file_name_status + '.npy'
        # map_numpy_save = np.save(filename_npy, map_data)
        # now = datetime.datetime.now()
        # status_now = [format(now)[0:19], hum_num, hum_spd_max, car_num, car_spd_max, prob_min, prob_min_pos[0], prob_min_pos[1], obj_ae_err, maha_nor, maha_abn, motion_err]
        # status_list = np.vstack((status_list, status_now))

        # np.save('output/TestbedStatusData_AbnTemp.npy', status_list)