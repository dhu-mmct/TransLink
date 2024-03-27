import numpy as np
from torch.utils.data import DataLoader

import dataset_change as dataset
import torch
from torch import nn

class Classifier(nn.Module):
    def __init__(self, cin):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(cin * 2, 256)  # out 256
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 2)     # out 64
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.1)
        self.fc3 = nn.Linear(64, 16)      # out 16
        self.relu = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(16, 2)       # out 4
        '''self.relu = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(4, 2)        # out 32
        self.relu = nn.ReLU(inplace=True)
        self.fc6 = nn.Linear(32, 16)        # out 16
        self.relu = nn.ReLU(inplace=True)
        self.fc7 = nn.Linear(16, 8)         # out 8
        self.relu = nn.ReLU(inplace=True)
        self.fc8 = nn.Linear(8, 4)          # out 4
        self.relu = nn.ReLU(inplace=True)
        self.fc9 = nn.Linear(4, 2)          # out 2'''

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)  # x (64, 512) x1,x2 (64, 256)
        x = self.fc1(x)  # x (64, 128)
        x = self.relu(x)
        x = self.fc2(x)
        '''x = self.relu(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)

        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.fc7(x)
        x = self.relu(x)
        x = self.fc8(x)
        x = self.relu(x)
        x = self.fc9(x)
'''
        return x



class AppearanceTrackletEmb(torch.nn.Module):
    def __init__(self, temporal_len=65, num_layer=4, device='cuda', stage='train', feedforward_dim=512, nheads=1,
                 dropout=0.1, pos_emb=False):
        super(AppearanceTrackletEmb, self).__init__()
        self.channel = 512
        self.input_channel = 2053
        self.num_layer = num_layer

        self.pooling = nn.AdaptiveAvgPool2d((512, 1))
        self.classifier = Classifier(512)
        # queue layer, key layer, value layer
        self.Q_layer = torch.nn.ModuleList([])
        self.K_layer = torch.nn.ModuleList([])
        self.V_layer = torch.nn.ModuleList([])
        for n in range(num_layer):
            if n == 0:
                self.Q_layer.append(torch.nn.Conv1d(self.input_channel, self.channel, 1))
                self.K_layer.append(torch.nn.Conv1d(self.input_channel, self.channel, 1))
                self.V_layer.append(torch.nn.Conv1d(self.input_channel, self.channel, 1))
            else:
                self.Q_layer.append(torch.nn.Conv1d(self.channel, self.channel, 1))
                self.K_layer.append(torch.nn.Conv1d(self.channel, self.channel, 1))
                self.V_layer.append(torch.nn.Conv1d(self.channel, self.channel, 1))

    def transform(self, Xv):
        for n in range(self.num_layer):
            prev_Xv = Xv
            #test Xv (1, 2053, 30)
            Xq = self.Q_layer[n](Xv)  # test Xv (1, 512, 30)
            Xk = self.K_layer[n](Xv)  #test Xv (1, 512, 30)
            Xv = torch.nn.functional.leaky_relu(self.V_layer[n](Xv)) #test Xv (1, 512, 30)
            # test Xv (1, 30, 30)
            att = torch.matmul(Xq.permute(0, 2, 1), Xk) / np.sqrt(self.channel)  # att (24, 65, 65)

            att = torch.nn.functional.softmax(att, dim=2)

            att = torch.unsqueeze(att, dim=2)
            Xv = torch.unsqueeze(Xv, dim=1)

            #test Xv(1, 30, 512)
            Xv = torch.sum(att * Xv, 3)  # Xv (24, 65, 512)
            # test Xv(1, 512, 30)
            Xv = Xv.permute(0, 2, 1)  # Xv (24, 512, 65)

            # skip
            if Xv.shape[1] == prev_Xv.shape[1]:
                Xv = Xv + prev_Xv

            # normalize
            # if n == self.num_layer - 1:
            #     Xv = torch.sum(Xv * scores, dim=2) / torch.sum(scores, dim=2) # Xv (24, 512)

        return Xv

    def forward(self, dataleft, dataright):
        # datanpy['appearance_embs'] (24, 2048, 65) now (64,2053,30)
        xl = dataleft
        xl = xl.type(torch.cuda.FloatTensor)
        xl = xl.permute(0, 2, 1)
        xr = dataright
        xr = xr.type(torch.cuda.FloatTensor)
        xr = xr.permute(0, 2, 1)

        xl = self.transform(xl)
        xr = self.transform(xr) #[1, 512, 30]

        xl = self.pooling(xl).squeeze(-1)  # [1, 512]
        xr = self.pooling(xr).squeeze(-1)

        y = self.classifier(xl, xr)  # [B, 2]
        if not self.training:
            y = torch.softmax(y, dim=1)
        return y


# if __name__ == '__main__':
#     # x1 = torch.ones((2, 1, 30, 3))
#     # x2 = torch.ones((2, 1, 30, 3))
#     data = dataset.LinkData(
#         root='/home/shuanghong/Downloads/github/dataset/MOT17/train',
#         mode='train'
#     )
#     dataloader = DataLoader(
#         dataset=data,
#         batch_size=2,
#         shuffle=True,
#         num_workers=1,
#         drop_last=False
#     )
#     print(len(data))
#     print(len(dataloader))
#     for i, (pair1, pair2, pair3, pair4, labels) in enumerate(dataloader):
#         pairs_1 = torch.cat((pair1[0], pair2[0], pair3[0], pair4[0]), dim=0)
#         pairs_2 = torch.cat((pair1[1], pair2[1], pair3[1], pair4[1]), dim=0)
#         label = torch.cat(labels, dim=0)
#         print(pairs_1.shape)
#         print(label)
#         # m = PostLinker()
#         y = m(pair1[0], pair1[1])
#         print(y)
#         break
