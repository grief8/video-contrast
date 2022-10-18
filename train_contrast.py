import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from data_loader import get_loader
from network.paraAttention import ParaAttention
import eval_tools

# configure training record
writer = SummaryWriter()
dataset = 'tvsum'
# load training and testing dataset
train_loader_list, test_dataset_list, data_file = get_loader("datasets/fcsn_{}.h5".format(dataset), "2D", 1)
# device use for training and testing
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# number of epoch to train
EPOCHS = 500
# array for calc. eval fscore
fscore_arr = np.zeros(len(train_loader_list))
kernel_size = (5, 1)
learning_rate = 1e-4
# model declaration
# model = FCSN_2D_sup()
# model = BiConvLSTM(1024, 320, kernel_size, 1, True, True, False)
model = ParaAttention(1024, 320, kernel_size, 1, True, True, False)
# model = ConvLSTM(1024, 320, kernel_size, 1, True, True, False)
# model = nn.LSTM(1024, 5, 1, bidirectional=True)
# optimizer declaration
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# switch to train mode
model.train()
# put model in to device
model.to(device)
print(device)
print('BiConvLSTM')
print('kernel_size: ', kernel_size)
print('learning_rate: ', learning_rate)
for i in range(len(train_loader_list)):
    for epoch in range(EPOCHS):
        for batch_i, (feature,label,_) in enumerate(train_loader_list[i]):
            feature = feature.unsqueeze(0).permute(1, 4, 2, 0, 3) #[5,1024,1,320] --> [5, 320, 1024, 1, 1]
            # feature = feature.permute(2, 0, 3, 1).squeeze(0)
            feature = feature.to(device)
            label = label.to(device) #[5,320]

            # reshape
            label = label.view(-1) #[5*320] every element indicates non-key or key (0 or 1)

            # loss criterion
            label_0 = 0.5*label.shape[0]/(label.shape[0]-label.sum())
            label_1 = 0.5*label.shape[0]/label.sum()
            weights = torch.tensor([label_0,label_1], dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=weights)

            # zero the parameter gradients
            optimizer.zero_grad()
            _, outputs = model(feature) # output shape [5,2,1,320] torch.Size([2, 5, 320, 1, 1])

            # reshape output
            # outputs = outputs.permute(0,3,2,1).contiguous().view(-1,2) #[5*320,2] 2 choices prob., non-key or key
            # outputs = outputs.squeeze(4).permute(1, 2, 0, 3).contiguous().view(-1, 2)
            # outputs = torch.stack(outputs[0])
            outputs = outputs.squeeze(4).permute(1, 2, 0, 3).contiguous().view(-1, 2)
            # outputs = outputs[0].contiguous().view(-1, 2)
            loss = criterion(outputs, label)
            loss.backward()
            # print grad
            # print(model.upscore16.weight.grad)
            optimizer.step()

        # eval every 5 epoch
        if(epoch+1) % 5 == 0:
            model.eval()
            eval_res_avg = [] # for all testing video results
            for feature,label,index in test_dataset_list[i]: # index has been +1 in dataloader.py
                feature = feature.view(1, 1,1024,1,-1).to(device) # [1024,1,320] -> [1,1024,1,320]
                # pred_score = model(feature).view(-1, 320)  # [1,2,1,320] -> [2,320]
                # feature = feature.view(1,1024,1,-1).unsqueeze(0).permute(1, 4, 2, 0, 3).to(device) # [1024,1,320] -> [1,1024,1,320]
                # feature = feature.permute(1, 2, 0).to(device)
                _, pred_score = model(feature)
                # pred_score = pred_score.squeeze(3).squeeze(3).view(-1, 320)
                # pred_score = torch.stack(pred_score[0])
                pred_score = pred_score.squeeze(3).view(-1, 320)
                # we only want key frame prob. -> [1]
                pred_score = torch.softmax(pred_score, dim=0)[1] # [320]

                video_name = "video_{}".format(index)
                video_info = data_file[video_name]

                # select key shots by video_info and pred_score
                # pred_summary: [N]
                _, _, _, pred_summary = eval_tools.select_keyshots(video_info, pred_score)
                true_summary_arr = video_info['user_summary'][()] # shape (20,N), summary from 20 users, each row is a binary vector

                eval_res = [eval_tools.eval_metrics(pred_summary, true_summary) for true_summary in true_summary_arr] # shape [20,3] 20 for users,3 for[precision, recall, fscore]
                eval_res = np.mean(eval_res, axis=0).tolist()  # for tvsum
                # eval_res = np.max(eval_res, axis=0).tolist()    # for summe
                eval_res_avg.append(eval_res) # [[precision1, recall1, fscore1], [precision2, recall2, fscore2]......]
                if eval_res[2] > 0.6:
                    np.savetxt(
                        'experiments/tvsum_b2/best_pred_core_' + video_name + '_split_{}_epoch_{:0>3d}.csv'.format(i, epoch),
                        pred_score.cpu().detach().numpy())
                    np.savetxt('experiments/tvsum_b2/best_' + video_name + '_split_{}_epoch_{:0>3d}.csv'.format(i, epoch),
                               np.array([pred_summary, true_summary_arr[0]]))

            np.savetxt('train_tvsum.csv', np.array(eval_res_avg))
            eval_res_avg = np.mean(eval_res_avg, axis=0).tolist()
            precision = eval_res_avg[0]
            recall = eval_res_avg[1]
            fscore = eval_res_avg[2]
            print("split:{} epoch:{:0>3d} precision:{:.1%} recall:{:.1%} fscore:{:.1%}".format(i, epoch, precision, recall, fscore))

            model.train()

            # store the last fscore for eval, and remove model from GPU
            if((epoch+1)==EPOCHS):
                fscore_arr[i] = fscore
                print("split:{} epoch:{:0>3d} precision:{:.1%} recall:{:.1%} fscore:{:.1%}".format(i, epoch, precision, recall, fscore))
                # release model from GPU
                model = model.cpu()
                torch.cuda.empty_cache()

            writer.add_scalar("bi-eval_{}_epoch/precision".format(dataset), precision, epoch,
                              time.time())  # tag, Y, X -> 當Y只有一個時
            writer.add_scalar("bi-eval_{}_epoch/recall".format(dataset), recall, epoch, time.time())
            writer.add_scalar("bi-eval_{}_epoch/fscore".format(dataset), fscore, epoch, time.time())
            


# print eval fscore
print("average fscore:{:.1%}".format(fscore_arr.mean()))