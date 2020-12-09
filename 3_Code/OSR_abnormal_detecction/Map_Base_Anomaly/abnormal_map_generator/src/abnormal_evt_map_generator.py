import glob
import cv2
import copy
import datetime
import numpy as np
import matplotlib.pyplot as plt
import osr_functions as ofn


def map_data_syn(map_data, data_length):
    idx_rand = int(data_length * np.random.rand(1))
    map_data_add1 = np.nan_to_num(np.load(map_files[idx_rand]))
    for iy in range(map_y_size):
        for ix in range(map_x_size):
            for iz in range(map_layer_num):
                if map_data[iy, ix, iz] == 0:
                    map_data[iy, ix, iz] = map_data_add1[iy, ix, iz]

    return map_data

global map_x_size, map_y_size, map_layer_num
map_x_size = 170
map_y_size = 140
map_layer_num = 12

ofn.prob_hist_initialize()
ofn.cnn_evt_initialize()
map_obs_prb = 1 - plt.imread("output/obs_prob_map.png")[:, :, 0]  # object probability report map
map_files_1 = sorted(glob.glob('mapDB_v2/moving_agent/npy/*.npy'), reverse=False)
map_files_2 = sorted(glob.glob('mapDB_v2/feedback/caution/npy/*.npy'), reverse=False)
map_files = map_files_1 + map_files_2
data_length = np.shape(map_files)[0]

for idx in range(data_length):
# for idx in range(100):
    map_data = np.nan_to_num(np.load(map_files[idx]))
    current_layered_map = map_data
    my, mx, mz = np.shape(current_layered_map)
    elv_info, thr_info = ofn.elv_thr_map_info(current_layered_map)
    elv_map = current_layered_map[:, :, 7] * np.clip(current_layered_map[:, :, 8], 0, 1)
    thr_map = current_layered_map[:, :, 9] * np.clip(current_layered_map[:, :, 10], 0, 1)
    elv_prob_map, thr_prob_map = ofn.elv_thr_prob_map_generation(current_layered_map)
    evt_obrv_map = np.dstack((100 * elv_map, 2.5 * thr_map, np.zeros((my, mx))))
    evt_prob_map = np.dstack((2.5 * elv_prob_map, 2.5 * thr_prob_map, 250 * np.ones((my, mx))))
    evt_map = np.hstack((evt_obrv_map, evt_prob_map))
    evt_qt_map = ofn.elv_thr_prob_map_reducer(ofn.elv_thr_prob_map_reducer(evt_map))
    filename_img = 'mapDB_v2/0/evt/evt_nor_' + str(idx) + '.bmp'
    cv2.imwrite(filename_img, evt_qt_map.astype(np.uint8)[:, :, ::-1])

    # Positions for abnormal add point
    map_generated = copy.deepcopy(current_layered_map)
    elv_update = np.clip(current_layered_map[:, :, 8], 0, 1)
    thr_update = np.clip(current_layered_map[:, :, 10], 0, 1)
    obs_pos = np.argwhere(elv_update * map_obs_prb > 0.8)

    # Add random abnormal elevation
    if np.shape(obs_pos)[0] > 0:
        print('Abnormal elevation and thermal map generation', idx, data_length, 'Elv', elv_info, 'Thr', thr_info, np.shape(obs_pos)[0])
        obs_num = np.clip(int(10 * np.random.rand(1)), 3, np.shape(obs_pos)[0])
        if np.shape(obs_pos)[0] > 2:
            for cnt in range(obs_num):
                map_generated[obs_pos[cnt][0], obs_pos[cnt][1], 7] = 1.5 * np.random.rand(1) + 1.0 # add 1.0~2.5m
                # map_generated[obs_pos[cnt][0], obs_pos[cnt][1], 7] = 2.5

            elv_map_gen = map_generated[:, :, 7] * np.clip(map_generated[:, :, 8], 0, 1)
            thr_map_gen = map_generated[:, :, 9] * np.clip(map_generated[:, :, 10], 0, 1)
            elv_prob_map_gen, thr_prob_map_gen = ofn.elv_thr_prob_map_generation(map_generated)
            evt_obrv_map_gen = np.dstack((100 * elv_map_gen, 2.5 * thr_map_gen, np.zeros((my, mx))))
            evt_prob_map_gen = np.dstack((2.5 * elv_prob_map_gen, 2.5 * thr_prob_map_gen, 250 * np.ones((my, mx))))
            evt_map_gen = np.hstack((evt_obrv_map_gen, evt_prob_map_gen))
            evt_qt_map_gen = ofn.elv_thr_prob_map_reducer(ofn.elv_thr_prob_map_reducer(evt_map_gen))

            filename_img = 'mapDB_v2/1/evt/evt_abn_' + str(idx) + '.bmp'
            cv2.imwrite(filename_img, evt_qt_map_gen.astype(np.uint8)[:, :, ::-1])


