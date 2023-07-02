import torch
from torch.autograd import Variable
import numpy as np

def attack_xyz(inputs, outputs, model, is_cuda, connectivity_dict,
            epsilon = 1e-2, iters = 50, epsilon_step = 1e-3, attack_range = 'all'):
    '''
    Generate a perturbation for the input motion sequence\n
    inputs: the clean input sequence\n
    output: the ground truth\n
    model: the target model\n
    is_cuda: the parameter deciding whether the function use GPU\n
    connectivity_dict: the connectivity dictionary of how the joints are connected in the pose\n
    epsilon: the boundary of the perturbation, default is 1e-2\n
    iters: the numbers of the iterations for the optimization of the perturbation, default is 50\n
    epsilon_step: the size of each step in an iteration, default is 1e-3\n
    attack_range: the frames need to be perturbed, default is 'all'\n
    (If you want to perturb specific frame, you can use a list like[0,1,2] instead)
    '''

    #Send the inputs and outputs to the device
    if is_cuda:
        inputs, outputs = Variable(inputs.cuda()).float(), Variable(outputs.cuda()).float()
        inputs.requires_grad = True
        outputs.requires_grad = True
    #Initiate
    output_n = outputs.data.shape[1]
    input_n = inputs.data.shape[1]
    data_len = output_n + input_n
    #Extract the scale of the input motion sequence
    scale = Scale_Function(inputs)
    #Initial the perturbation
    perturbation = torch.rand_like(inputs).type_as(inputs).cuda() * epsilon * scale
    #If the victim frames are not selected, attack all the frames in the input motion sequence
    if attack_range == 'all':
        attack_range = range(input_n)
    #Turn the perutrbation against the unattacked frames to zero
    for frame in range(perturbation.shape[1]):
        if frame in attack_range:
            continue
        else:
            perturbation[:, frame] *= 0

    #Optimization begins
    for i in range(iters):
        #Add perturbation to the clean sequence
        pert_inputs = inputs + perturbation
        pert_inputs = Variable(pert_inputs).float()
        pert_inputs.requires_grad = True
        if is_cuda:
            pert_inputs = pert_inputs.cuda()
        #Do the same operation to the perturbed sequence as the original model does
        pad_idx = np.repeat([input_n - 1], output_n)
        i_idx = np.append(np.arange(0, input_n), pad_idx)

        #Forward pass the inputs through the model
        model.zero_grad()
        out, p = model(pert_inputs[:, i_idx, :, :])

        #Calculate the loss
        loss_pred = torch.mean(torch.norm(out - outputs, p=2, dim=3, keepdim=True))#Prediction loss
        loss_bl = Bone_Length_Loss(inputs, pert_inputs, connectivity_dict)#Bone length loss
        loss_temp = temp_Loss(inputs, pert_inputs, 2)#Temporal loss
        loss = loss_pred - 0.5 * (loss_temp + loss_bl)
        #Backward propagation to get the gradient of the perturbed input
        loss.backward()
        data_grad = pert_inputs.grad.data
        #Turn the unattacked frames' gradients to zero
        for frame in range(data_grad.shape[1]):
            if frame in attack_range:
                continue
            else:
                data_grad[:, frame] *= 0
        #Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        #Optimize the perturbed data by adjusting each coordinate in the poses of the input data
        perturbation += epsilon_step*scale*sign_data_grad
        perturbation = torch.clamp(perturbation, -epsilon*scale, epsilon*scale)

    return perturbation

def Scale_Function(input_3d):
    input_n = input_3d.data.shape[1]
    if input_3d.shape[-1] != 3:
        inputs_temp = input_3d.clone().view(input_3d.data.shape[0], input_n, -1, 3)
        x_dif = torch.min(torch.max(inputs_temp[:,:,:,0], dim = 2)[0] - torch.min(inputs_temp[:,:,:,0], dim = 2)[0]).item()
        y_dif = torch.min(torch.max(inputs_temp[:,:,:,1], dim = 2)[0] - torch.min(inputs_temp[:,:,:,1], dim = 2)[0]).item()
        z_dif = torch.min(torch.max(inputs_temp[:,:,:,2], dim = 2)[0] - torch.min(inputs_temp[:,:,:,2], dim = 2)[0]).item()
        scale = min(x_dif,y_dif,z_dif)
    else:
        x_dif = torch.min(torch.max(input_3d[:,:,:,0], dim = 2)[0] - torch.min(input_3d[:,:,:,0], dim = 2)[0]).item()
        y_dif = torch.min(torch.max(input_3d[:,:,:,1], dim = 2)[0] - torch.min(input_3d[:,:,:,1], dim = 2)[0]).item()
        z_dif = torch.min(torch.max(input_3d[:,:,:,2], dim = 2)[0] - torch.min(input_3d[:,:,:,2], dim = 2)[0]).item()
        scale = min(x_dif,y_dif,z_dif)
    return scale

def Get_Bone_Length(pose_3d, connectivity_dict):
    batch_len, frame_len, _, _ = pose_3d.shape
    limb_num = len(connectivity_dict)
    bl = torch.zeros([batch_len, frame_len, limb_num]).cuda()
    bl_vec = torch.zeros([batch_len, frame_len, limb_num, 3]).cuda()
    #Calculate the bone lengths of each limb in the sequence by the connectivity dictionary
    for limb in range(limb_num):
        bl_vec[:, :, limb, :] = pose_3d[:, :, connectivity_dict[limb][0], :] - pose_3d[:, :, connectivity_dict[limb][1], :]
    bl = bl_vec.norm(p=2, dim=3)
    return bl

def Bone_Length_Loss(clean_3d, pert_3d, connectivity_dict):
    batch_len, frame_len, _, _ = clean_3d.shape
    limb_num = len(connectivity_dict)
    bl_clean = torch.zeros([batch_len, frame_len, limb_num]).cuda()
    bl_pert = torch.zeros([batch_len, frame_len, limb_num]).cuda()
    dif = torch.zeros([batch_len, frame_len, limb_num]).cuda()
    bl_clean = Get_Bone_Length(clean_3d, connectivity_dict)
    bl_pert = Get_Bone_Length(pert_3d, connectivity_dict)
    dif = bl_pert - bl_clean
    loss_bl = torch.norm(dif)

    return loss_bl

def Derivative_Spatial(spatial_seq):
    frame_len = spatial_seq.shape[1]
    previous_seq = spatial_seq[:, 0:frame_len-2, :, :]
    later_seq = spatial_seq[:, 1:frame_len-1, :, :]
    derivative_seq = later_seq - previous_seq
    return derivative_seq

def temp_Loss(clean_3d, pert_3d, max_n):
    loss_s = 0
    target_seq = clean_3d
    target_pert = pert_3d
    b = 1/max_n
    for i in range(max_n):
        der_clean = Derivative_Spatial(target_seq)
        der_pert = Derivative_Spatial(target_pert)
        v_dif = der_pert - der_clean
        loss_s += b * torch.norm(v_dif)
        target_seq = der_clean
        target_pert = der_pert
    return loss_s