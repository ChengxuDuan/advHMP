from torch.utils.data import Dataset
import numpy as np
from utils import data_utils


class CMU_Motion3D(Dataset):

    def __init__(self, path_to_data, actions, input_n=10, output_n=10, split=0, data_mean=0, data_std=0, dim_used=0):

        self.path_to_data = path_to_data
        self.split = split
        actions = data_utils.define_actions_cmu(actions)
        # actions = ['walking']
        
        if split == 0:
            path_to_data = path_to_data + '/train/'
            is_test = False
        else:
            path_to_data = path_to_data + '/test/'
            is_test = True
        all_seqs, dim_ignore, dim_use, data_mean, data_std = data_utils.load_data_cmu_3d(path_to_data, actions,
                                                                                         input_n, output_n,
                                                                                         data_std=data_std,
                                                                                         data_mean=data_mean,
                                                                                         is_test=is_test)
        if not is_test:
            dim_used = dim_use
        joints_order = np.array([35,34,32,37,31,30,
                                26,25,23,28,22,21,
                                19,18,17,15,14, 
                                9,10,11,12,
                                3,4,5,6,
                                0,1,2,7,8,13,
                                16,20,29,24,27,33,36])
        
        n, frame_n, _ = all_seqs.shape
        self.all_seqs = all_seqs.reshape(n, frame_n, -1, 3)[:, :, joints_order, :]
        self.dim_used = dim_used
        self.input_seqs = all_seqs.reshape(n, frame_n, -1, 3)[:, :input_n, joints_order, :]
        self.output_seqs = all_seqs.reshape(n, frame_n, -1, 3)[:, input_n:, joints_order, :]
        self.data_mean = data_mean
        self.data_std = data_std

    def __len__(self):
        return np.shape(self.input_seqs)[0]

    def __getitem__(self, item):
        return self.input_seqs[item], self.output_seqs[item], self.all_seqs[item]
