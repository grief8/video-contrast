import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from data_loader import get_loader
from dereplace import *
import eval_tools

dataset = 'tvsum'
# load training and testing dataset
train_loader_list, test_dataset_list, data_file = get_loader("datasets/fcsn_{}.h5".format(dataset), "2D", 1)
# device use for training and testing
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = torch.load('tvsum.pt')
model.to(device)

for feature, label, index in test_dataset_list[0]:
    feature = feature.permute(1, 2, 0).to(device)
    pred_score, out_feats, att_score = model(feature)
    pred_score = pred_score.view(-1, 320)
    pred_score = torch.softmax(pred_score, dim=0)[1]

    video_name = "video_{}".format(index)
    video_info = data_file[video_name]

    _, _, _, pred_summary = eval_tools.select_keyshots(video_info, pred_score)
    true_summary_arr = video_info['user_summary'][()]

    eval_res = [eval_tools.eval_metrics(pred_summary, true_summary) for true_summary in
                true_summary_arr]
    print(eval_res)