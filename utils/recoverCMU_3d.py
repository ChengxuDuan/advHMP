import numpy as np
import torch
import pdb

def recoverCMU_3d(gt,pred):
	joint_to_ignore = np.array([31, 32, 33, 34, 35, 36, 37])
	joint_equal = np.array([15, 15, 15, 8, 8, 2, 2])
	unchange_joint=np.array([25, 26, 27, 28, 29, 30])  
	tem=torch.zeros([gt.shape[0],gt.shape[1],len(joint_to_ignore)+len(unchange_joint),gt.shape[-1]]).cuda()
	#pdb.set_trace()
	pred_3d=torch.cat([pred, tem], dim=2)
	pred_3d[:,:,joint_to_ignore]=pred_3d[:,:,joint_equal]
	pred_3d[:,:,unchange_joint]=gt[:,:,unchange_joint]

	return pred_3d
