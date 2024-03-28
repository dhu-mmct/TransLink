"""
@Author: Du Yunhao
@Filename: AFLink.py
@Contact: dyh_bupt@163.com
@Time: 2021/12/28 19:55
@Discription: Appearance-Free Post Link
"""
import os
from collections import defaultdict
from os.path import join, exists

import glob
import numpy as np
from scipy.optimize import linear_sum_assignment

import AFLink.config as cfg
import torch
from AFLink.dataset_change import LinkData
#for window 30
from AFLink.model_change import AppearanceTrackletEmb
import time


INFINITY = 1e5


class AFLink:
    def __init__(self, path_in, path_out, model, dataset, thrT: tuple, thrS: int, thrP: float):
        self.thrP = thrP  # 预测阈值
        self.thrT = thrT  # 时域阈值
        self.thrS = thrS  # 空域阈值
        self.model = model  # 预测模型
        self.dataset = dataset  # 数据集类
        self.path_out = path_out  # 结果保存路径
        # self.track = np.loadtxt(path_in, delimiter=',')
        self.track = np.load(path_in)
        self.model.cuda()
        self.model.eval()

    # 获取轨迹信息
    def gather_info(self):
        id2info = defaultdict(list)
        self.track = self.track[np.argsort(self.track[:, 0])]  # 按帧排序
        for row in self.track:
            f, i = row[:2]
            id2info[i].append(np.concatenate((row[:1], row[2:6], row[10:])))
        self.track = np.array(self.track)
        id2info = {k: np.array(v) for k, v in id2info.items()}
        return id2info

    # 损失矩阵压缩
    def compression(self, cost_matrix, ids):
        # 行压缩 （取出每行的最小值）
        mask_row = cost_matrix.min(axis=1) < self.thrP
        matrix = cost_matrix[mask_row, :]
        ids_row = ids[mask_row]
        # 列压缩 （取出每列的最小值）
        mask_col = cost_matrix.min(axis=0) < self.thrP
        matrix = matrix[:, mask_col]
        ids_col = ids[mask_col]
        # 矩阵压缩
        return matrix, ids_row, ids_col

    # 连接损失预测
    def predict(self, track1, track2):
        x1, x2 = track1[:, :5], track2[:, :5]
        x1, x2 = self.dataset.transform(x1, x2)
        det1, det2 = track1[:, 5:], track2[:, 5:]
        det1 = self.dataset.fill_or_cut(det1, True)
        det2 = self.dataset.fill_or_cut(det2, False)
        track1 = np.concatenate((x1, det1), axis=1)
        track2 = np.concatenate((x2, det2), axis=1)  # track1 (1,30,5)

        track1 = torch.tensor(track1, dtype=torch.float)
        track2 = torch.tensor(track2, dtype=torch.float)

        # track1 = track1.permute(1, 0)
        # track2 = track2.permute(1, 0)
        #track1, track2 = track1.unsqueeze(0).cuda(), track2.unsqueeze(0).cuda()  # track1 (1,1,30,5)

        #for window_size 30
        track1 = track1.reshape(1, 30, 2053)
        track2 = track2.reshape(1, 30, 2053)
        # for window_size 65
        # track1 = track1.reshape(1, 65, 2053)
        # track2 = track2.reshape(1, 65, 2053)

        score = self.model(track1, track2)[0, 1].detach().cpu().numpy()
        return 1 - score

    # 去重复: 即去除同一帧同一ID多个框的情况
    @staticmethod
    def deduplicate(tracks):
        _, index = np.unique(tracks[:, :2], return_index=True, axis=0)  # 保证帧号和ID号的唯一性
        return tracks[index]

    def transform(self, x1, x2):
        # min-max normalization
        min_ = np.concatenate((x1, x2), axis=0).min(axis=0)
        max_ = np.concatenate((x1, x2), axis=0).max(axis=0)
        subtractor = (max_ + min_) / 2
        divisor = (max_ - min_) / 2 + 1e-5
        x1 = (x1 - subtractor) / divisor
        x2 = (x2 - subtractor) / divisor
        # unsqueeze channel=1
        x1 = x1.unsqueeze(dim=0)
        x2 = x2.unsqueeze(dim=0)
        return x1, x2

    # 主函数
    def link(self):
        id2info = self.gather_info()
        for i in id2info:
            print(len(id2info[i]))
        num = len(id2info)  # 目标数量
        ids = np.array(list(id2info))  # 目标ID
        fn_l2 = lambda x, y: np.sqrt(x ** 2 + y ** 2)  # L2距离
        cost_matrix = np.ones((num, num)) * INFINITY  # 损失矩阵
        '''计算损失矩阵'''
        for i, id_i in enumerate(ids):  # 前一轨迹
            for j, id_j in enumerate(ids):  # 后一轨迹
                if id_i == id_j: continue  # 禁止自娱自乐
                info_i, info_j = id2info[id_i], id2info[id_j]
                fi, bi = info_i[-1][0], info_i[-1][1:3]
                fj, bj = info_j[0][0], info_j[0][1:3]
                if not self.thrT[0] <= fj - fi < self.thrT[1]: continue
                if self.thrS < fn_l2(bi[0] - bj[0], bi[1] - bj[1]): continue

                cost = self.predict(info_i, info_j)
                if cost <= self.thrP: cost_matrix[i, j] = cost
        '''二分图最优匹配'''
        id2id = dict()  # 存储临时匹配结果
        ID2ID = dict()  # 存储最终匹配结果
        cost_matrix, ids_row, ids_col = self.compression(cost_matrix, ids)
        indices = linear_sum_assignment(cost_matrix)
        for i, j in zip(indices[0], indices[1]):
            if cost_matrix[i, j] < self.thrP:
                id2id[ids_row[i]] = ids_col[j]
        for k, v in id2id.items():
            if k in ID2ID:
                ID2ID[v] = ID2ID[k]
            else:
                ID2ID[v] = k
        # print('  ', ID2ID.items())
        '''结果存储'''
        res = self.track.copy()
        for k, v in ID2ID.items():
            res[res[:, 1] == k, 1] = v
        res = self.deduplicate(res)
        #fmt = '%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d' + 2048 * '%d'
        np.savetxt(self.path_out, res[:,:10], fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')

if __name__ == '__main__':

    dir_in = '/mnt/disk/shuanghong/new_dataset_result_wqx/reid-deepsort'

    dir_out = dir_in + '-translink-test'
    if not exists(dir_out): os.mkdir(dir_out)
    model = AppearanceTrackletEmb()

    cfg.model_savedir = '/home/shuanghong/Downloads/github/project/strongsortPure/data_space/models'
    model.load_state_dict(torch.load(join(cfg.model_savedir, 'newmodel_0.0464.pth')))   #best:newmodel_0.0464.pth
    dataset = LinkData(cfg.root_train, 'train')
    start = time.time()
    for path_in in sorted(glob.glob(dir_in + '/*.npy')):
        print('processing the file {}'.format(path_in))
        dir_save = join(dir_out, path_in.split('/')[-1][:-4] + '.txt')
        linker = AFLink(
            path_in=path_in,
            path_out=dir_save,
            model=model,
            dataset=dataset,
            thrT=(-10, 30),  # (0,30) or (-10,30)  icassp (-10, 30)
            thrS=80,  # 75   icassp: 80
            thrP=0.4,  # 0.05 or 0.10     0.4(icassp的结果都是0.4)
        )
        linker.link()