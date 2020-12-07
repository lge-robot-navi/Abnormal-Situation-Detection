import glob
import pickle
import os
import cv2
import numpy as np
from time import sleep
import osr_functions as ofn
import matplotlib.pyplot as plt

global map_x_size, map_y_size, map_layer_num
map_x_size = 170
map_y_size = 140
map_layer_num = 6

# update_temp = plt.imread("output/map_update_fix123_v5.png")[:,:,0] # update map adjust
while 1:
    prob_hist = [[[0 for _ in range(map_layer_num)] for _ in range(map_x_size)] for _ in range(map_y_size)]
    for x in range(map_x_size):
        for y in range(map_y_size):
            prob_hist[y][x][0] = np.zeros(9).astype(np.uint32)  # Human Density Histogram 0 ~ 8 people
            prob_hist[y][x][1] = np.zeros(9).astype(np.uint32)  # Human Direction Histogram 0:stop, 1~8
            prob_hist[y][x][2] = np.zeros(9).astype(np.uint32)  # Car Density Histogram: 0 ~ 8 cars
            prob_hist[y][x][3] = np.zeros(9).astype(np.uint32)  # Car Direction Histogram 0:stop, 1~8
            prob_hist[y][x][4] = np.zeros(5).astype(np.uint32)  # Elevation Histogram 0 ~ 2.5m --> 0 ~ 4, 0.5m / step
            prob_hist[y][x][5] = np.zeros(10).astype(np.uint32) # Temperature Histogram 0 ~ 100'C --> 0 ~ 9, 10'C / step

    map_files_fix = sorted(glob.glob('mapDB_v2/0/npy/*.npy'), reverse=True)
    map_files_mov = sorted(glob.glob('mapDB_v2/moving_agent/npy/*.npy'), reverse=False)
    map_files_fdb = sorted(glob.glob('mapDB_v2/feedback/caution/npy/*.npy'), reverse=False)
    print('fix', np.shape(map_files_fix), 'mov', np.shape(map_files_mov), 'fdb', np.shape(map_files_fdb))
    # map_files = map_files_fix[0:10000] + map_files_mov + map_files_fdb
    map_files = map_files_fix + map_files_mov + map_files_fdb
    data_length = np.shape(map_files)[0]

    for idx in range(data_length):
        map_data = np.load(map_files[idx])
        map_data[np.isnan(map_data)] = 0
        current_layered_map = map_data
        hum_num, hum_den, hum_spd, hum_dir, car_num, car_den, car_spd, car_dir = ofn.map_den_spd_dir(current_layered_map)
        elv_info, thr_info = ofn.elv_thr_map_info(current_layered_map)

        if np.shape(current_layered_map)[2] == 12:
            print('Prob. Calc.', idx, data_length, 'Hum', hum_num, 'Car', car_num, 'Elv', elv_info, 'Thr', thr_info)
            obj_update = current_layered_map[:, :, 11]
            elv_update = current_layered_map[:, :, 8]
            thr_update = current_layered_map[:, :, 10]

            for x in range(0, map_x_size):
                for y in range(0, map_y_size):
                    # Human and car probability calculation
                    if obj_update[y, x] > 0:
                        if hum_den[y, x] == 0:
                            prob_hist[y][x][0][0] += 1
                        else:
                            den_tmp = np.clip(hum_den[y, x], 0, 8)
                            dir_tmp = np.clip(hum_dir[y, x], 0, 8)
                            for idx_y in range(-2, 3):
                                for idx_x in range(-2, 3):
                                    y_tmp = np.clip(y + idx_y, 0, map_y_size)
                                    x_tmp = np.clip(x + idx_x, 0, map_x_size)
                                    prob_hist[y_tmp][x_tmp][0][den_tmp] += 1
                                    prob_hist[y_tmp][x_tmp][1][dir_tmp] += 1
                        if car_den[y, x] == 0:
                            prob_hist[y][x][2][0] += 1
                        else:
                            den_tmp = np.clip(car_den[y, x], 0, 8)
                            dir_tmp = np.clip(car_dir[y, x], 0, 8)
                            for idx_y in range(-2, 3):
                                for idx_x in range(-2, 3):
                                    y_tmp = np.clip(y + idx_y, 0, map_y_size)
                                    x_tmp = np.clip(x + idx_x, 0, map_x_size)
                                    prob_hist[y_tmp][x_tmp][2][den_tmp] += 1
                                    prob_hist[y_tmp][x_tmp][3][dir_tmp] += 1

                    # Elevation probability calculation
                    if elv_update[y, x] > 0:
                        elv_idx = int(np.clip(current_layered_map[y, x, 7]/0.5, 0, 4))
                        for idx_y in range(-2, 3):
                            for idx_x in range(-2, 3):
                                y_tmp = np.clip(y + idx_y, 0, map_y_size)
                                x_tmp = np.clip(x + idx_x, 0, map_x_size)
                                prob_hist[y_tmp][x_tmp][4][elv_idx] += 1

                    # Thermal probability calculation
                    if thr_update[y, x] > 0:
                        thr_idx = int(np.clip(np.clip(current_layered_map[y, x, 9], 0, 100)/10, 0, 9))
                        for idx_y in range(-2, 3):
                            for idx_x in range(-2, 3):
                                y_tmp = np.clip(y + idx_y, 0, map_y_size)
                                x_tmp = np.clip(x + idx_x, 0, map_x_size)
                                prob_hist[y_tmp][x_tmp][5][thr_idx] += 1

    with open('output/probability_weight_all', 'wb') as fp:
        pickle.dump(prob_hist, fp)

    # Object probability report map
    object_prob = np.zeros((map_y_size, map_x_size))
    for x in range(map_x_size):
        for y in range(map_y_size):
            obj_cnt = np.sum(prob_hist[y][x][0][1:])
            total_cnt = np.sum(prob_hist[y][x][0][:])
            no_cnt = np.sum(prob_hist[y][x][0][0])
            if no_cnt > obj_cnt:
                object_prob[y, x] = obj_cnt / (total_cnt + 1)

    print(np.max(object_prob))
    object_prob[object_prob>0.1]=0
    print(np.max(object_prob))
    object_prob = np.sqrt(np.clip(100*object_prob,  0, 1))
    object_prob_save = 255 * np.dstack((object_prob, object_prob, object_prob))
    filename = 'output/obj_prob_map.png'
    cv2.imwrite(filename, object_prob_save.astype(np.uint8)[:, :, ::-1])

    # Obstacle probability report map
    obs_tmp = np.ones((map_y_size, map_x_size))
    for x in range(map_x_size):
        for y in range(map_y_size):
            no_obs_cnt = prob_hist[y][x][4][0]
            all_cnt = np.sum(prob_hist[y][x][4][:])
            if no_obs_cnt > 0:
                obs_tmp[y, x] = np.sqrt(np.clip(1 - no_obs_cnt / (all_cnt + 1), 0, 1))

    obs_prob_save = 250 * np.dstack((obs_tmp, obs_tmp, obs_tmp))
    filename = 'output/obs_prob_map.png'
    cv2.imwrite(filename, obs_prob_save.astype(np.uint8)[:, :, ::-1])

    print('Probability weight and report map saved. Wait for 10 min.')
    # sleep(600)
