from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def read_tensorboard_data(tensorboard_path, val_name):
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    print(ea.scalars.Keys())
    val = ea.scalars.Items(val_name)
    return val

def gen_xy(path, val_name):
    val = read_tensorboard_data(path, val_name)
    x = [i.step for i in val]
    y = [j.value for j in val]
    return x,y

class local_scale:
    def __init__(self, ax, N):
        self.axins = inset_axes(ax, width="30%", height="30%", loc=7,
                           bbox_to_anchor=(0, 0, 1, 1),
                           bbox_transform=ax.transAxes)
        self.ax = ax
        self.ys = []
        self.x = np.array(list(range(N)))

    def plot_data(self, x, y, line=None):
        if line is not None:
            self.axins.plot(x, y, line)
        else:
            self.axins.plot(x, y)
        self.ys.append(y)

    def apply_scale(self):
        # 设置放大区间
        xlim0 = 27
        xlim1 = 29.5
        ylim0 = 0.78
        ylim1 = 0.85

        # 调整子坐标系的显示范围
        self.axins.set_xlim(xlim0, xlim1)
        self.axins.set_ylim(ylim0, ylim1)

        # 建立父坐标系与子坐标系的连接线
        # loc1 loc2: 坐标系的四个角
        # 1 (右上) 2 (左上) 3(左下) 4(右下)
        mark_inset(self.ax, self.axins, loc1=2, loc2=1, fc="none", ec='k', lw=1)
        self.axins.grid(True)

if __name__ == "__main__":
    rootPath = "/home/hans/WorkSpace/Data/Text/Models/Transcriptor/FL_CRNN/"
    val_list = ['test_acc/IIIT.5K_global_acc', 'test_acc/SVT_global_acc', 'test_acc/ICDAR2015_global_acc']
    # val_list = ['test_loss/IIIT.5K_global_loss', 'test_loss/SVT_global_loss', 'test_loss/ICDAR2015_global_loss']

    new_val_list=['test_acc/IIIT.5K/IIIT5K_test_20190829.json_global_acc',
                  'test_acc/SVT/svt_test_20190828_cropped.json_global_acc',
                  'test_acc/ICDAR2015/ICDAR2015_test_20190829.json_global_acc']
    idx = 1
    val_list = [val_list[idx]]
    new_val_list = [new_val_list[idx]]

    N = 30
    fig, ax = plt.subplots(1, 1)
    plt.xlabel('Round')
    plt.ylabel(val_list[0].split('/')[1])
    mini_ax = local_scale(ax, N)

    for val_name in val_list:
        # oEeB = "FedAvg/20191028-171809-1E800B/events.out.tfevents.1572288554.CSJD-Server"
        # x1,y1=gen_xy(rootPath+oEeB, val_name)
        # oEeB_2 = "FedAvg/20191029-220850-1E800B/events.out.tfevents.1572391032.CSJD-Server"
        # x_2, y_2 = gen_xy(rootPath + oEeB_2, val_name)
        # x1 += [i + 25 for i in x_2]
        # y1 += [j for j in y_2]

        oEeB = "FedBoost/20200116-130043-1E800B_FedBoost/events.out.tfevents.1579184927.cspcvision"
        x1, y1 = gen_xy(rootPath + oEeB, val_name)

        ax.plot(x1[0:N], y1[0:N], label="FedAvg")
        mini_ax.plot_data(x1[0:N], y1[0:N])

    for val_name in new_val_list:
        fedboost_he = "FedAvg_HE/20200714-224956-1E800B_FedAvg_HE/events.out.tfevents.1594770577.CSJD-Server"
        x2, y2 = gen_xy(rootPath + fedboost_he, val_name)
        ax.plot(x2[0:N], y2[0:N],'--', label="FedAvg with HE")
        mini_ax.plot_data(x2[0:N], y2[0:N], '--')

    # for val_name in val_list:
    #     fedavg = "FedBoost/20200204-114127-1E800B_FedBoost_NewProportion/events.out.tfevents.1580821973.cspcvision"
    #     x1,y1=gen_xy(rootPath+fedavg, val_name)
    #     ax.plot(x1[0:N], y1[0:N], label="FedBoost")
    #     mini_ax.plot_data(x1[0:N], y1[0:N])
    #
    # for val_name in new_val_list:
    #     fedavg = "FedBoost_HE/20200711-214522-1E800B_FedBoost_HE/events.out.tfevents.1594513088.CSJD-Server"
    #     x2,y2=gen_xy(rootPath+fedavg, val_name)
    #     ax.plot(x2[0:N], y2[0:N],'g--', label="FedBoost with HE")
    #     mini_ax.plot_data(x2[0:N], y2[0:N],'g--')
    #
    #
    # for val_name in new_val_list:
    #     fedboost_he = "FedBoost_HE/20200623-145913-1E800B_FedBoost_HE_unified_initial_weights/events.out.tfevents.1592935806.CSJD-Server"
    #     x3, y3 = gen_xy(rootPath + fedboost_he, val_name)
    #     ax.plot(x3[0:N], y3[0:N],'r--', label="FedBoost with HE and DP")
    #     mini_ax.plot_data(x3[0:N], y3[0:N], 'r--')

    mini_ax.apply_scale()
    ax.legend()
    ax.grid(True)
    plt.show()