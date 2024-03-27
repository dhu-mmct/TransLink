
import numpy as np
from os.path import join
from random import randint, normalvariate
from torch.utils.data import Dataset, DataLoader

import AFLink.config as cfg

#MOT17
SEQ = {
    'train': [
        'MOT17-02-FRCNN',
        'MOT17-04-FRCNN',
        'MOT17-05-FRCNN',
        'MOT17-09-FRCNN',
        'MOT17-10-FRCNN',
        'MOT17-11-FRCNN',
        'MOT17-13-FRCNN',
        'MOT17-02-DPM',
        'MOT17-04-DPM',
        'MOT17-05-DPM',
        'MOT17-09-DPM',
        'MOT17-10-DPM',
        'MOT17-11-DPM',
        'MOT17-13-DPM',
        'MOT17-02-SDP',
        'MOT17-04-SDP',
        'MOT17-05-SDP',
        'MOT17-09-SDP',
        'MOT17-10-SDP',
        'MOT17-11-SDP',
        'MOT17-13-SDP',
    ],
    # 'val': [
    #     'MOT17-09-FRCNN',
    # ],
    'test': [
        'MOT17-01-FRCNN',
        'MOT17-03-FRCNN',
        'MOT17-06-FRCNN',
        'MOT17-07-FRCNN',
        'MOT17-08-FRCNN',
        'MOT17-12-FRCNN',
        'MOT17-14-FRCNN'
    ]
}

# SEQ = {
#     'train': [
#         'MOT20-01',
#         'MOT20-02',
#         'MOT20-03'
#     ],
#     # 'val': [
#     #     'MOT17-09-FRCNN',
#     # ],
#     # 'test': [
#     #     'MOT17-01-FRCNN',
#     #     'MOT17-03-FRCNN',
#     #     'MOT17-06-FRCNN',
#     #     'MOT17-07-FRCNN',
#     #     'MOT17-08-FRCNN',
#     #     'MOT17-12-FRCNN',
#     #     'MOT17-14-FRCNN'
#     # ]
# }


class LinkData(Dataset):
    def __init__(self, root, mode='train', minLen=cfg.model_minLen, inputLen=cfg.model_inputLen):
        """
        :param minLen: 仅考虑长度超过该阈值的GT轨迹
        :param inputLen: 网络输入轨迹长度
        """
        self.minLen = minLen
        self.inputLen = inputLen
        if root:
            assert mode in ('train', 'val')
            self.root = root
            self.mode = mode
            self.id2info, self.detect_list= self.initialize()
            self.ids = list(self.id2info.keys())

    def initialize(self):
        id2info = dict()
        detect_list = dict()

        for seqid, seq in enumerate(SEQ['train'], start=1):
            path_gt = join(self.root, '{}/gt/gt.txt'.format(seq))
            path_imgs = join(self.root, '{}/img1'.format(seq))
            detect_feat = np.load(join('/mnt/disk/shuanghong/MOT17/reid', '{}.npy'.format(seq)))
            # detect_feat = np.load(join('/home/shuanghong/Downloads/github/dataset/MOT20/train-reid', '{}.npy'.format(seq)))

            gts = np.loadtxt(path_gt, delimiter=',')
            gts = gts[(gts[:, 6] == 1) * (gts[:, 7] == 1)]  # 仅考虑“considered" & "pedestrian"
            detect_feat = np.concatenate((detect_feat[:,:2],detect_feat[:,9:]),axis=1)
            ids = set(gts[:, 1])
            for objid in ids:
                id_ = objid + seqid * 1e5
                track = gts[gts[:, 1] == objid]
                det_tra = detect_feat[detect_feat[:,1] == objid]
                det_tra = np.concatenate((det_tra[:,:1],det_tra[:,2:]), axis=1)
                fxywh = [[t[0], t[2], t[3], t[4], t[5]] for t in track]
                if len(fxywh) >= self.minLen:
                    id2info[id_] = np.array(fxywh)
                    detect_list[id_] = np.array(det_tra)
        return id2info,detect_list

    def fill_or_cut(self, x, former: bool):
        """
        :param x: input
        :param former: True代表该轨迹片段为连接时的前者
        """
        lengthX, widthX = x.shape
        if lengthX >= self.inputLen:
            if former:
                x = x[-self.inputLen:]
            else:
                x = x[:self.inputLen]
        else:
            zeros = np.zeros((self.inputLen - lengthX, widthX))
            if former:
                x = np.concatenate((zeros, x), axis=0)
            else:
                x = np.concatenate((x, zeros), axis=0)
        return x

    def transform(self, x1, x2):
        # fill or cut
        x1 = self.fill_or_cut(x1, True)
        x2 = self.fill_or_cut(x2, False)
        # min-max normalization
        min_ = np.concatenate((x1, x2), axis=0).min(axis=0)
        max_ = np.concatenate((x1, x2), axis=0).max(axis=0)
        subtractor = (max_ + min_) / 2
        divisor = (max_ - min_) / 2 + 1e-5
        x1 = (x1 - subtractor) / divisor
        x2 = (x2 - subtractor) / divisor

        return x1, x2


    def extraItem(self, item):
        info = self.id2info[self.ids[item]]
        det = self.detect_list[self.ids[item]]
        # all frames of one id
        numFrames = min(info.shape[0], det.shape[0])

        idxCut = randint(self.minLen // 3, numFrames - self.minLen // 3)  # 随机裁剪点
        idxCut_add = idxCut + int(normalvariate(-5, 3))
        idxCut_min = idxCut + int(normalvariate(5, 3))
        '''while idxCut_min <= idxCut_add:
            idxCut_add = idxCut + int(normalvariate(-8, 1))
            idxCut_min = idxCut + int(normalvariate(8, 1))'''
        # 样本对儿 move front or back for about 5
        info1 = info[:idxCut_add]  # 为索引添加随机偏差
        info2 = info[idxCut_min:]  # 为索引添加随机偏差
        det1 = det[:idxCut_add]
        det2 = det[idxCut_min:]

        # 返回正负样本对儿
        return info1, info2,det1,det2

    def __getitem__(self, item):
        # if item >= len(self.ids):
        #     item = randint(0,len(self.ids))
        info1, info2, det1, det2 = self.extraItem(item)
        num = randint(0,len(self.ids)-1)
        while num == item:
            num = randint(0, len(self.ids)-1)
        # print(num)
        info3, info4, det3, det4 = self.extraItem(num)

        det3 = self.fill_or_cut(det3, True)[:,1:]
        det4 = self.fill_or_cut(det4, False)[:,1:]
        det1 = self.fill_or_cut(det1, True)[:,1:]
        det2 = self.fill_or_cut(det2, False)[:,1:]
        info1, info2 = self.transform(info1, info2)
        info3, info4 = self.transform(info3, info4)

        data1 = np.concatenate((info1, det1), axis=1)
        data2 = np.concatenate((info2, det2), axis=1)
        data3 = np.concatenate((info3, det3), axis=1)
        data4 = np.concatenate((info4, det4), axis=1)
        return  (data1, data2),\
                (data3, data4),\
                (data1, data3),\
                (data1, data4),\
                (1,1,0,0)


    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':
    dataset = LinkData(
        root='/home/shuanghong/Downloads/github/dataset/MOT17/train',
        mode='train'
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        drop_last=False
    )
    print(len(dataset))
    print(len(dataloader))
    for i, (pair1, pair2, pair3, pair4, labels) in enumerate(dataloader):
        print(1)

        print(pair1)
        print(labels)
        break