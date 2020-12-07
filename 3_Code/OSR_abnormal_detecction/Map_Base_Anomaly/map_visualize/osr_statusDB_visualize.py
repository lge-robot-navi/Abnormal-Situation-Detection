import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
import time


statusDB_files = sorted(glob.glob('output/statusDB/*.npy'))
DB_size = np.shape(statusDB_files)[0]
print('Total DB : ', DB_size)
data_plot = -10 * np.ones((24*60*60, 5))

for idx_DB in range(DB_size):
    status_list = np.load(statusDB_files[idx_DB])[1:]
    data_length = np.shape(status_list)[0]
    print(statusDB_files[idx_DB], data_length)
    for idx_data in range(data_length):
        now = status_list[idx_data][0]
        month = now[5:7]
        date = now[8:10]
        try:
            hour = int(now[11:13])
            minute = int(now[14:16])
            second = int(now[17:])
            tic =  (hour * 60 + minute) * 60 + second
            data_plot[tic:tic+10000, :] = -10

            hum_num = int(status_list[idx_data, 1])
            car_num = int(status_list[idx_data, 3])
            prob = float(int(10*np.clip(6 - np.float32(status_list[idx_data, 5]), 0, 6)))/10
            ae = float(int(10*np.clip(np.float32(status_list[idx_data, 8]), 0, 6)))/10
            nor_dist = np.clip(np.float32(status_list[idx_data, 9]), 0, 20) + 0.01
            abn_dist = np.clip(np.float32(status_list[idx_data, 10]), 0, 20) + 0.01
            maha_ratio = np.clip(float(int(10*np.clip(np.log10(nor_dist / abn_dist), -3, 3)))/10 + 1, 0, 6)

            data_plot[tic, 0] = hum_num
            data_plot[tic, 1] = car_num
            data_plot[tic, 2] = prob
            data_plot[tic, 3] = ae
            data_plot[tic, 4] = maha_ratio

        except:
            print('time conversion error')

    try:
        style.use('fivethirtyeight')
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        time_hour = np.arange(0, 24, 1.0/3600)

        car_plot = ax1.scatter(time_hour, data_plot[:, 1], color="g", s=10)
        hum_plot = ax1.scatter(time_hour, data_plot[:, 0], color="r", s=10)
        ax1.legend((hum_plot, car_plot), ('Human', 'Car'), scatterpoints=1, loc='upper left', ncol=2, fontsize=15)

        prob_plot = ax2.scatter(time_hour, data_plot[:, 2], color="r", s=10)
        ae_plot = ax2.scatter(time_hour, data_plot[:, 3], color="g", s=10)
        maha_plot = ax2.scatter(time_hour, data_plot[:, 4], color="b", s=10)
        ax2.legend((prob_plot, ae_plot, maha_plot), ('Probability', 'Unusual', 'Feedback'), scatterpoints=1, loc='upper left', ncol=3, fontsize=15)

        ax1.set_xlim([0, 25])
        ax1.xaxis.set_ticks(np.arange(0, 25, 4))
        ax1.set_ylim([-0.5, 8])
        ax1.set_ylabel('object Number', size=15)
        ax1.hold(True)

        ax2.set_xlim([0, 25])
        ax2.xaxis.set_ticks(np.arange(0, 25, 4))
        ax2.set_ylim([-0.5, 5.5])
        ax2.set_xlabel('hour', size=15)
        ax2.set_ylabel('scores', size=15)
        ax2.hold(True)

        ax1.text(18, 6, "2020-{}-{}".format(month, date), fontsize=20, color='black')

        plt.show(block=False)
        figure_name = 'output/statusDB/figures/2020_'+ month + '_' + date + '_' + str(hour) + '.png'
        plt.savefig(figure_name, dpi=300)
        # time.sleep(0.5)
        plt.close()

    except:
        print('plotting error')

