import pickle as pkl
from os import walk
from torch.utils.data import Dataset
import numpy as np
from utils import data_utils

from matplotlib import pyplot as plt


class Pose3dPW3D(Dataset):

    def __init__(self, path_to_data, input_n=20, output_n=10, split=0):
        """

        :param path_to_data:
        :param input_n:
        :param output_n:
        :param split:
        """
        self.path_to_data = path_to_data
        self.split = split

        # since baselines (http://arxiv.org/abs/1805.00655.pdf and https://arxiv.org/pdf/1705.02445.pdf)
        # use observed 50 frames but our method use 10 past frames in order to make sure all methods are evaluated
        # on same sequences, we first crop the sequence with 50 past frames and then use the last 10 frame as input
        if split == 1:
            their_input_n = 50
        else:
            their_input_n = input_n
        seq_len = their_input_n + output_n

        if split == 0:
            self.data_path = path_to_data + '/train/'
        elif split == 1:
            self.data_path = path_to_data + '/test/'
        elif split == 2:
            self.data_path = path_to_data + '/validation/'
        all_seqs = []
        files = []
        for (dirpath, dirnames, filenames) in walk(self.data_path):
            files.extend(filenames)
        for f in files:
            with open(self.data_path + f, 'rb') as f:
                data = pkl.load(f, encoding='latin1')
                joint_pos = data['jointPositions']
                for i in range(len(joint_pos)):
                    seqs = joint_pos[i]
                    seqs = seqs - seqs[:, 0:3].repeat(24, axis=0).reshape(-1, 72)
                    n_frames = seqs.shape[0]
                    fs = np.arange(0, n_frames - seq_len + 1)
                    fs_sel = fs
                    for j in np.arange(seq_len - 1):
                        fs_sel = np.vstack((fs_sel, fs + j + 1))
                    fs_sel = fs_sel.transpose()
                    seq_sel = seqs[fs_sel, :]
                    if len(all_seqs) == 0:
                        all_seqs = seq_sel
                    else:
                        all_seqs = np.concatenate((all_seqs, seq_sel), axis=0)

        joints_order = [23,21,19,17,14,13,16,18,20,22,15,12,9,6,3,2,5,8,11,1,4,7,10,0]
        all_seqs = all_seqs[:, (their_input_n - input_n):, :]
        n, frame, joints_n = all_seqs.shape
        all_seqs = all_seqs.reshape(n, frame, -1, 3)
        
        self.all_seqs = all_seqs[:, :, joints_order, :]
        self.input_seq = all_seqs[:, :input_n, joints_order, :]
        self.output_seq = all_seqs[:, input_n:, joints_order, :]

    def __len__(self):
        return np.shape(self.input_seq)[0]

    def __getitem__(self, item):
        return self.input_seq[item], self.output_seq[item], self.all_seqs[item]
