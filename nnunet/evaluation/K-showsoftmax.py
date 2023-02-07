# a matplotlib test
import numpy as np
import scipy.io as sio
import random
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r'E:\Windows\Fonts\times.ttf')
id='130'

def draw():
    # # 定义热图的横纵坐标
    # xLabel = ['A', 'B', 'C', 'D', 'E']
    # yLabel = ['1', '2', '3', '4', '5']

    # # 准备数据阶段，利用random生成二维数据（5*5）
    # data = []
    # for i in range(5):
    #     temp = []
    #     for j in range(5):
    #         k = random.randint(0, 100)
    #         temp.append(k)
    #     data.append(temp)

    data = np.load(
        r'D:\Keenster\Projects\nnUnet-base\DATASET\nnUNet_raw\nnUNet_raw_data\Task501_Study\inferTs\merged\Study_'+id+'.npz')
    sm=data['softmax']
    # 作图阶段
    fig = plt.figure()
    # 定义画布为1*1个划分，并在第1个位置上进行作图
    ax = fig.add_subplot(111)
    # 定义横纵坐标的刻度
    # ax.set_yticks(range(len(yLabel)))
    # ax.set_yticklabels(yLabel, fontproperties=font)
    # ax.set_xticks(range(len(xLabel)))
    # ax.set_xticklabels(xLabel)
    # 作图并选择热图的颜色填充风格，这里选择hot
    label = 3
    slice = 17
    # im = ax.imshow(sm[label,slice,:,:], cmap=plt.cm.viridis)
    im = ax.imshow(sm[label, slice, :, :], cmap=plt.cm.gray)
    # 增加右侧的颜色刻度条
    plt.colorbar(im)
    # 增加标题
    plt.title('probablity map of label '+str(label)+', slice: '+str(slice), fontproperties=font)
    # show
    plt.show()

    # keenster add for softmax save
    sio.savemat(
        r'D:\Keenster\Projects\nnUnet-base\DATASET\nnUNet_raw\nnUNet_raw_data\Task501_Study\inferTs\merged\Study_'+id+'.mat',
        data)


d = draw()

# matplotlib色系可見https://matplotlib.org/stable/gallery/color/colormap_reference.html

from nnunet.inference.segmentation_export import save_softmax_nifti_from_softmax




# save_softmax_nifti_from_softmax(r"D:\Keenster\Projects\nnUnet-base\DATASET\nnUNet_raw\nnUNet_raw_data\Task501_Study\inferTs\merged\Study_126.npz", r"D:\Keenster\Projects\nnUnet-base\DATASET\nnUNet_raw\nnUNet_raw_data\Task501_Study\inferTs\merged\Study_126_softmax.nii.gz")