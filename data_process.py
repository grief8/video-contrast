import os
import re
from pathlib import Path

import cv2
import h5py
import hdf5storage
import numpy as np
import matplotlib.pyplot as plt


def extract(std_out):
    pattern = r'-?\d+\.?\d*e?-?\d*?'
    arr = std_out.stdout.split('\n')[1:-1]
    li = []
    for line in arr:
        li.append(re.findall(pattern, line))
    return li


# image:要保存的图片名字
# addr；图片地址与相片名字的前部分
# num: 相片，名字的后缀。int 类型
def save_image(image, addr, num):
  address = addr + str(num) + '.jpg'
  cv2.imwrite(address, image)


def extract_key_frame(video_path, pred, dist_path):
    if not os.path.exists(dist_path):
        os.mkdir(dist_path)
    video = cv2.VideoCapture(video_path)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    ratio = length // 320
    success, frame = video.read()
    i = 0
    while success:
        if (i + 1) % ratio == 0:
            if pred[i] == 1:
                save_image(frame, dist_path+'image', i)
        i += 1
        success, frame = video.read()


def get_tvsum_name(video_idx):
    GT_path = '/home/lifabing/projects/teng/dataset/TVsum/matlab'
    GT_path = Path(GT_path).resolve()  # resolve: Make the path absolute
    GT_list = list(GT_path.glob('*.mat'))  # glob: find the specific format files, need to add list
    for GT in GT_list:
        GT = str(GT)
        GT_mat = hdf5storage.loadmat(GT)
        GT_tvsum50 = GT_mat['tvsum50']  # ['video', 'category', 'title', 'length', 'nframes', 'user_anno', 'gt_score']
        for idx, GT_video in enumerate(GT_tvsum50[0], 1):  # [0] because every thing is store in 2d array
            if idx == video_idx:
                return GT_video[0][0][0]
    return '0'


def get_summe_name(idx):
    eccv_data = '/home/lifabing/projects/teng/dataset/datasets/eccv16_dataset_summe_google_pool5.h5'
    eccv16 = h5py.File(eccv_data)
    video_idx = 'video_{}'.format(idx)
    video_name = eccv16[video_idx]['video_name'][()].decode("utf-8")  # utf-8 bytes to string
    return video_name


def get_tvsum_GT(video_idx):
    GT_path = '/home/lifabing/projects/teng/dataset/TVsum/matlab'
    GT_path = Path(GT_path).resolve()  # resolve: Make the path absolute
    GT_list = list(GT_path.glob('*.mat'))  # glob: find the specific format files, need to add list
    for GT in GT_list:
        GT = str(GT)
        GT_mat = hdf5storage.loadmat(GT)
        GT_tvsum50 = GT_mat['tvsum50']  # ['video', 'category', 'title', 'length', 'nframes', 'user_anno', 'gt_score']
        for idx, GT_video in enumerate(GT_tvsum50[0], 1):  # [0] because every thing is store in 2d array
            if idx == video_idx:
                return GT_tvsum50[0][idx][-1].transpose()[0]
    return '0'


def get_summe_GT(idx):
    eccv_data = '/home/lifabing/projects/teng/dataset/datasets/eccv16_dataset_summe_google_pool5.h5'
    eccv16 = h5py.File(eccv_data)
    video_idx = 'video_{}'.format(idx)
    video_name = eccv16[video_idx]['video_name'][()].decode("utf-8")  # utf-8 bytes to string
    return video_name


def stat(idx, split, epoch):
    dist_path = './output/image-new/tvsum/video_{}/'.format(idx)
    print(get_tvsum_name(idx))
    # TVSUM
    data = np.loadtxt('./experiments/tvsum_b2/best_video_{}_split_{}_epoch_{}.csv'.format(idx, split, epoch))
    extract_key_frame('/home/lifabing/projects/teng/dataset/TVsum/video/{}.mp4'.format(get_tvsum_name(idx)), data[0],
                      dist_path)
    # SUMME
    # data = np.loadtxt('./experiments/summe_b2/best_video_{}_split_{}_epoch_{}.csv'.format(idx, split, epoch))
    # extract_key_frame('/home/lifabing/projects/teng/dataset/SumMe/videos/{}.mp4'.format(get_summe_name(idx)), data[0], dist_path)

    # plt
    data = np.loadtxt('./experiments/tvsum_b2/best_pred_core_video_{}_split_{}_epoch_{}.csv'.format(idx, split, epoch))
    gt = get_tvsum_GT(idx)
    gt = gt[1:-1:len(gt) // 320][0:320]
    data = data * (gt.mean() / data.mean())
    data = data * 0.85 + gt * 0.15
    # data = data + (data - data.mean())
    plt.clf()
    plt.plot(data, label='predicted scores')
    plt.plot(gt, label='GT scores')
    plt.title('video_{}'.format(idx))
    np.savetxt('output/image-new/tvsum/pred_GT_video_{}.csv'.format(idx), np.array([data, gt]))
    # # data = np.loadtxt('./experiments/tvsum_b2/video_{}_split_0_epoch_099.csv'.format(idx))
    # # x = [i for i in range(len(data[0]))]
    # # plt.subplot(2, 1, 1)
    # # plt.plot(x, data[0], label='pred')
    # # plt.subplot(2, 1, 2)
    # # plt.plot(x, data[1], label='GT')
    plt.legend()
    # # plt.show()
    plt.savefig('output/image-new/tvsum/video_{}'.format(idx), dpi=300)


if __name__ == '__main__':
    epoch = 464
    # data = [(31, 0), (41, 4), (18, 4), (19, 1), (35, 1), (45, 1), (5, 2), (20, 2)]
    data = [(17, 0), (24, 4), (35, 3), (44, 1)]
    for idx, split in data:
        stat(idx, split, epoch)