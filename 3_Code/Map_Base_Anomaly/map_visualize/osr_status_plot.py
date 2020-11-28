import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np

style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(4,1,1)
ax2 = fig.add_subplot(4,1,2)
ax3 = fig.add_subplot(4,1,3)
ax4 = fig.add_subplot(4,1,4)

# ['data-time', 'hum_num', 'car_num', 'aty_obj_status', 'y', 'x', 'aty_evt_status', 'y', 'x', 'max_temp', 'y', 'x', 'min_temp', 'mean_temp']

def animate(i):
    try:
        status_list = np.load('output/TestbedStatusHistory_v1.npy')
        # status_list = np.load('output/TestbedStatusData_v2.npy')
        # status_list = np.load('output/TestbedStatusData_AbnTemp.npy')
        print(status_list[-1, 0], np.shape(status_list))
        time_hour = range(-(np.shape(status_list)[0]-1), 0)

        ax1.clear()
        ax1.set_xlim([-50000, 100])
        ax1.set_ylim([-0.2, 3])
        ax1.set_ylabel('Status', size=12)
        ax1.hold(True)
        aty_obj_status = np.float32(status_list[1:,3])
        aty_evt_status = np.float32(status_list[1:,6])
        aty_obj_status_plot = ax1.scatter(time_hour, aty_obj_status+0.05, color="r", s=5)
        aty_evt_status_plot = ax1.scatter(time_hour, aty_evt_status-0.05, color="g", s=5)
        aty_obj_status_plot_now = ax1.scatter(0, aty_obj_status[-1]+0.05, edgecolors="black", color="r", s=100)
        aty_evt_status_plot_now = ax1.scatter(0, aty_evt_status[-1]-0.05, edgecolors="black", color="g", s=100)
        ax1.legend((aty_obj_status_plot, aty_evt_status_plot), ('Hum/Car', 'Elv/Thr'), scatterpoints=1, loc='upper left', ncol=4, fontsize=12)

        ax2.clear()
        ax2.set_xlim([-50000, 100])
        ax2.set_ylim([-0.2, 16])
        ax2.set_ylabel('Numbers', size=12)
        ax2.hold(True)
        hum_num = np.clip(np.float32(status_list[1:,1]), 0, 15) + 0.05
        car_num = np.clip(np.float32(status_list[1:,2]), 0, 15) - 0.05
        car_plot = ax2.scatter(time_hour, car_num, color="g", s=5)
        hum_plot = ax2.scatter(time_hour, hum_num, color="r", s=5)
        car_plot_now = ax2.scatter(0, car_num[-1], edgecolors="black", color="g", s=100)
        hum_plot_now = ax2.scatter(0, hum_num[-1], edgecolors="black", color="r", s=100)
        ax2.legend((hum_plot, car_plot), ('Hum', 'Car'), scatterpoints=1, loc='upper left', ncol=3, fontsize=12)

        ax3.clear()
        ax3.set_xlim([-50000, 100])
        ax3.set_ylim([-5, 100])
        ax3.set_ylabel('Temp "C', size=10)
        ax3.hold(True)
        max_temp = 2*np.clip(np.float32(status_list[1:,9]), 0, 100)
        min_temp = 2*np.clip(np.float32(status_list[1:,12]), 0, 100)
        mean_temp = 2*np.clip(np.float32(status_list[1:,13]), 0, 100)
        max_temp_plot = ax3.scatter(time_hour, max_temp, color="r", s=5)
        max_temp_now = ax3.scatter(0, max_temp[-1], edgecolors="black", color="r", s=100)
        mean_temp_plot = ax3.scatter(time_hour, mean_temp, color="g", s=5)
        mean_temp_now = ax3.scatter(0, mean_temp[-1], edgecolors="black", color="g", s=100)
        min_temp_plot = ax3.scatter(time_hour, min_temp, color="b", s=5)
        min_temp_now = ax3.scatter(0, min_temp[-1], edgecolors="black", color="b", s=100)
        ax3.legend((max_temp_plot, mean_temp_plot, min_temp_plot), ('Max', 'Med', 'Min'), scatterpoints=1, loc='upper left', ncol=3, fontsize=12)

        ax4.clear()
        ax4.imshow(np.flipud(plt.imread("output/testbed_170x140_v9.bmp")/255.0))
        ax4.set_xlim([0, 170])
        ax4.set_ylim([0, 140])
        ax4.set_aspect('equal')
        ax4.hold(True)
        pos_y_obj = np.float32(status_list[1:,4])
        pos_x_obj = np.float32(status_list[1:,5])
        pos_y_evt = np.float32(status_list[1:,7])
        pos_x_evt = np.float32(status_list[1:,8])
        pos_y_maxT = np.float32(status_list[1:,10])
        pos_x_maxT = np.float32(status_list[1:,11])

        pos_0_obj = ax4.scatter(pos_x_obj, 140 - pos_y_obj, color=[0.1, 0.1, 0])
        pos_0_evt = ax4.scatter(pos_x_evt, 140 - pos_y_evt, color=[0.1, 0.1, 0])
        pos_0_maxT = ax4.scatter(pos_x_maxT, 140 - pos_y_maxT, color=[0.1, 0.1, 0])

        pos_1_obj = ax4.scatter(pos_x_obj[-3600:], 140 - pos_y_obj[-3600:], color=[0.2, 0, 0])
        pos_1_evt = ax4.scatter(pos_x_evt[-3600:], 140 - pos_y_evt[-3600:], color=[0, 0.2, 0])
        pos_1_maxT = ax4.scatter(pos_x_maxT[-3600:], 140 - pos_y_maxT[-3600:], color=[0, 0, 0.2])

        pos_2_obj = ax4.scatter(pos_x_obj[-300:], 140 - pos_y_obj[-300:], color=[0.4, 0, 0])
        pos_2_evt = ax4.scatter(pos_x_evt[-300:], 140 - pos_y_evt[-300:], color=[0, 0.4, 0])
        pos_2_maxT = ax4.scatter(pos_x_maxT[-300:], 140 - pos_y_maxT[-300:], color=[0, 0, 0.4])

        pos_3_obj = ax4.scatter(pos_x_obj[-30:], 140 - pos_y_obj[-30:], color=[0.7, 0, 0])
        pos_3_evt = ax4.scatter(pos_x_evt[-30:], 140 - pos_y_evt[-30:], color=[0, 0.7, 0])
        pos_3_maxT = ax4.scatter(pos_x_maxT[-30:], 140 - pos_y_maxT[-30:], color=[0, 0, 0.7])

        pos_4_evt = ax4.scatter(pos_x_evt[-1], 140 - pos_y_evt[-1], color=[0, 1, 0])
        pos_4_obj = ax4.scatter(pos_x_obj[-1], 140 - pos_y_obj[-1], color=[1, 0, 0])
        pos_4_maxT = ax4.scatter(pos_x_maxT[-1], 140 - pos_y_maxT[-1], color=[0, 0, 1])

        ax4.legend((pos_4_obj, pos_4_evt, pos_4_maxT), ('Hum/Car', 'Elv/Thr', 'MaxTemp'), scatterpoints=1, loc='upper left', ncol=3, fontsize=8)
    except:
        print("plotting error")

ani = animation.FuncAnimation(fig, animate, interval=5000)
plt.show()