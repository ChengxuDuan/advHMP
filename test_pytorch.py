import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
from args import parser
import torch.nn as nn
import os
from time import strftime, localtime
import pandas as pd
from progress.bar import Bar

from TrajCNN import TrajectoryNet
from utils import recoverh36m_3d
from utils import recoverCMU_3d
from utils import data_utils
from utils.h36motion3d import H36motion3D
from utils.cmu_motion_3d import CMU_Motion3D
from utils.pose3dpw3d import Pose3dPW3D
import Attack_3d as Attack

connectivity_dict = {"H36M":[[0, 2, 1], [1, 2, 1], [2, 3, 1], [3, 4, 1], [4, 12, 1],#left arm
                             [5, 12, 0], [5, 6, 0], [6, 7, 0], [7, 8, 0], [7, 9, 0],#right arm
                              [10, 11, 0], [11, 12, 0], [12, 13, 0],#torso
                              [14, 13, 0],  [14, 15, 0], [15, 16, 0], [16, 17, 0],#left leg
                               [18, 13, 1],[18, 19, 1],[19, 20, 1], [20, 21, 1]]#right leg
                    ,"CMU":[[0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 4, 1], [4, 5, 1], [5, 15, 1],#left arm
                             [6, 7, 0], [7, 8, 0], [8, 9, 0], [9, 10, 0], [10, 11, 0], [11, 15, 0],#right arm
                              [12, 13, 0], [13, 14, 0], [14, 15, 0], [15, 16, 0],#torso
                              [16, 17, 1],  [17, 18, 1], [18, 19, 1], [19, 20, 1],#left leg
                               [16, 21, 0],[21, 22, 0],[22, 23, 0], [23, 24, 0]]#right leg
                    ,"3DPW":[[11, 5, 0], [5, 6, 0], [6, 7, 0], [7, 8, 0], [8, 9, 0],  # left arm
                                [14, 19, 0], [19, 20, 0], [20, 21, 0], [21, 22, 0], # left leg
                                [10, 11, 0], [11, 12, 0], [12, 13, 0], [13, 14, 0], # torso
                                [0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 4, 1], [4, 11, 1], # right arm
                                [14, 15, 1], [15, 16, 1], [16, 17, 1], [17, 18, 1], # right leg
                                ]}

def main():
    assert args.dataset in ["H36M", "CMU", "3DPW"]
    print(">>> loading data")
    input_n = args.input_len
    output_n = args.seq_len - args.input_len
    
    # load datasets for testing
    test_data = dict()
    if args.dataset=="H36M":
        acts = data_utils.define_actions(args.actions)
        for act in acts:
            test_dataset = H36motion3D(
                path_to_data=args.data_dir,
                actions=act,
                input_n=input_n,
                output_n=output_n,
                split=1,
            )
            test_data[act] = DataLoader(
                dataset=test_dataset,
                batch_size=args.test_batch,
                shuffle=False,
                num_workers=args.job,
                pin_memory=True,
            )
    elif args.dataset=="CMU":
        acts = data_utils.define_actions_cmu(args.actions)
        train_dataset = CMU_Motion3D(
            path_to_data=args.data_dir_cmu,
            actions=args.actions,
            input_n=input_n,
            output_n=output_n,
            split=0
        )
        data_std = train_dataset.data_std
        data_mean = train_dataset.data_mean
        dim_used = train_dataset.dim_used
        for act in acts:
            test_dataset = CMU_Motion3D(
                path_to_data=args.data_dir_cmu,
                actions=act,
                input_n=input_n,
                output_n=output_n,
                split=1,
                data_mean=data_mean,
                data_std=data_std,
                dim_used=dim_used,
            )
            test_data[act] = DataLoader(
                dataset=test_dataset,
                batch_size=args.test_batch,
                shuffle=False,
                num_workers=args.job,
                pin_memory=True,
            )
    elif args.dataset=="3DPW":
        test_dataset = Pose3dPW3D(
            path_to_data=args.data_dir_3dpw,
            input_n=input_n,
            output_n=output_n,
            split=1,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.test_batch,
            shuffle=False,
            num_workers=args.job,
            pin_memory=True,
        )

    print(">>> data loaded !")
    print(">>> test data {}".format(test_dataset.__len__()))
    
    #Initiate the model
    model = TrajectoryNet(
        input_n = input_n,
        output_n = output_n,
        dropout = args.dropout).to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.95, 0.9995))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    #Load the pretrained model
    epoch_pretrained = 0
    model_path_len = os.path.join(args.checkpoint, args.dataset, args.dataset+'_TrajectoryCNN_short.pth')
    checkpoint = torch.load(model_path_len)
    epoch_pretrained = checkpoint["epoch_pretrained"]
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    print("Model loaded")
    
    print(
        ">>> total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters()) / 1000000.0
        )
    )
    #Test
    ret_log = np.array([epoch_pretrained + 1])
    head = np.array(["epoch"])

    if args.dataset == "3DPW":
        print("Evaluate clean sequences start.")
        test_3d,_,_,_ = test(model, test_loader, is_attack=False)
        ret_log = np.append(ret_log, test_3d.cpu().numpy()*1000)
        if output_n == 15:
            head = np.append(head, ["1003d", "2003d", "3003d", "4003d", "5003d"])
        elif output_n == 30:
            head = np.append(
                head,
                [
                    "1003d",
                    "2003d",
                    "3003d",
                    "4003d",
                    "5003d",
                    "6003d",
                    "7003d",
                    "8003d",
                    "9003d",
                    "10003d",
                ],
            )
    else:
        for act in acts:
            print("Evaluate clean "+act+" start.")
            test_3d,_,_,_ = test(model, test_data[act], is_attack=False, action=act)
            ret_log = np.append(ret_log, test_3d.cpu().numpy())
            head = np.append(
                head, [act + "3d80", act + "3d160", act + "3d320", act + "3d400"]
            )
            if output_n > 10:
                head = np.append(head, [act + "3d560", act + "3d1000"])

    print("==========Attack Starts==========")
    LBL = 0
    Lvel = 0
    Lacc = 0
    if args.dataset == "3DPW":
        test_3d_adv, avg_LBL, avg_vel, avg_acc = test(model, test_loader,
                                                           is_attack=True, action=None, is_cuda=True,
                                                           connectivity_dict=connectivity_dict[args.dataset],
                                                             epsilon=args.epsilon, iters=args.iters,
                                                               epsilon_step=args.epsilon_step, attack_range=args.attack_range)
        LBL = LBL + avg_LBL*1000
        Lvel = Lvel + avg_vel*1000
        Lacc = Lacc + avg_acc*1000
        if output_n == 15:
            head = np.append(head, ["1003d_adv", "2003d_adv", "3003d_adv", "4003d_adv", "5003d_adv"])
        elif output_n == 30:
            head = np.append(
                head,
                [
                    "1003d_adv",
                    "2003d_adv",
                    "3003d_adv",
                    "4003d_adv",
                    "5003d_adv",
                    "6003d_adv",
                    "7003d_adv",
                    "8003d_adv",
                    "9003d_adv",
                    "10003d_adv",
                ],
            )
        ret_log = np.append(ret_log, test_3d_adv.cpu().numpy()*1000)
    else:
        for act in acts:
            print("Perturb "+act+" sequences done.")
            test_3d_adv, avg_LBL, avg_vel, avg_acc = test(model, test_data[act],
                                                           is_attack=True, action=act, is_cuda=True,
                                                           connectivity_dict=connectivity_dict[args.dataset],
                                                             epsilon=args.epsilon, iters=args.iters,
                                                               epsilon_step=args.epsilon_step, attack_range=args.attack_range)
            LBL = LBL + avg_LBL
            Lvel = Lvel + avg_vel
            Lacc = Lacc + avg_acc
            ret_log = np.append(ret_log, test_3d_adv.cpu().numpy())
            head = np.append(
                head, [act + "3d80_adv", act + "3d160_adv", act + "3d320_adv", act + "3d400_adv"]
            )
            if output_n > 10:
                head = np.append(head, [act + "3d560_adv", act + "3d1000_adv"])
        LBL = LBL/len(acts)
        Lvel = Lvel/len(acts)
        Lacc = Lacc/len(acts)
    print("Average Bone length Change: ",LBL,"\nAvearage velocity change in joints: ",Lvel,"\nAvearage acceleration change in joints: ", Lacc)
    df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
    df.to_csv("./"+args.dataset+"_TrajCNN_Rec.csv", header=head, index=False)


def test(model, test_loader, is_attack=False, action=None, is_cuda=True, connectivity_dict=None,
          epsilon = 1e-2, iters = 50, epsilon_step = 1e-3, attack_range = 'all'):
    mpjpe = torch.zeros((1, args.seq_len - args.input_len)).to(device)
    
    if args.dataset == "H36M" or args.dataset == "CMU":
        if args.seq_len - args.input_len == 25:
            eval_frame = [1, 3, 7, 9, 13, 24]
        elif args.seq_len - args.input_len == 10:
            eval_frame = [1, 3, 7, 9]
    elif args.dataset == "3DPW":
        if args.seq_len - args.input_len == 15:
            eval_frame = [2, 5, 8, 11, 14]
        elif args.seq_len - args.input_len == 30:
            eval_frame = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]

    if is_attack:
        mpjpe_x , avg_LBL, avg_vel, avg_acc= evaluate(model, test_loader, is_attack, action, is_cuda,
                                                       connectivity_dict, epsilon, iters, epsilon_step, attack_range)
        mpjpe += mpjpe_x
        return mpjpe[0][eval_frame], avg_LBL, avg_vel, avg_acc
    else:
        mpjpe_x,_,_,_ = evaluate(model, test_loader, is_attack, action, is_cuda)
        mpjpe += mpjpe_x
        return mpjpe[0][eval_frame], 0, 0, 0

def evaluate(model, test_loader, is_attack = False, action = None, is_cuda = True,
              connectivity_dict = None, epsilon = 1e-2, iters = 50, epsilon_step = 1e-3, attack_range = 'all'):
    model.eval()
    N = 0
    if is_attack:
        total_LBL = 0
        total_vel = 0
        total_acc = 0
        if len(attack_range)==0:
            attack_range='all'
    mpjpe1 = torch.zeros((1, args.seq_len - args.input_len)).to(device)

    bar = Bar(">>>", fill=">", max=len(test_loader))
    start_time = time.time()
    for batch_id, (input_seq, output_seq, load_data) in enumerate(test_loader):
        batch_time=time.time()
        n = load_data.shape[0]
        load_data = load_data.float().to(device)
        input_seq = input_seq.float().to(device)
        output_seq = output_seq.float().to(device)

        gth = load_data[:, args.input_len:, :, :]
        input_seq = input_seq[:, :, :args.joints_input, :]
        if is_attack:
            perturbation = Attack.attack_xyz(input_seq, output_seq[:, :, :args.joints_input, :], model, is_cuda, connectivity_dict, epsilon, iters, epsilon_step, attack_range)
            pert_input = input_seq + perturbation
            last_frame = pert_input[:, args.input_len - 1, :]
            last_frame = last_frame.unsqueeze(1)
            last_frame_repeat = last_frame.repeat(1, args.seq_len - args.input_len, 1, 1)
            pert_input = torch.cat((pert_input, last_frame_repeat), 1)
            pert_input = pert_input.to(device)
            out, p = model(pert_input)
        else:
            last_frame = input_seq[:, args.input_len - 1, :]
            last_frame = last_frame.unsqueeze(1)
            last_frame_repeat = last_frame.repeat(1, args.seq_len - args.input_len, 1, 1)
            input_data = torch.cat((input_seq, last_frame_repeat), 1)
            input_data = input_data.to(device)
            out, p = model(input_data)

        if args.dataset == "H36M":
            out = recoverh36m_3d.recoverh36m_3d(gth, out)

        elif args.dataset == "CMU":
            out = recoverCMU_3d.recoverCMU_3d(gth, out)

        if not os.path.exists('./seq'):
            os.makedirs(os.path.join('./seq'))

        if action != None:
            np.save('./seq/truth_seq_'+args.dataset+'_'+action+'.npy',load_data.cpu().detach().numpy())
            if is_attack:
                if args.dataset == "H36M":
                    pert_save = recoverh36m_3d.recoverh36m_3d(load_data[:, :args.input_len], pert_input[:, :args.input_len])
                elif args.dataset == "CMU":
                    pert_save = recoverCMU_3d.recoverCMU_3d(load_data[:, :args.input_len], pert_input[:, :args.input_len]) 
                elif args.dataset == "3DPW":
                    pert_save = pert_input[:, :args.input_len]
                np.save('./seq/pert_seq_'+args.dataset+'_'+action+'.npy',
                        torch.cat((pert_save, out), 1).cpu().detach().numpy())
            else:
                np.save('./seq/clean_seq_'+args.dataset+'_'+action+'.npy',
                        torch.cat((load_data[:, :args.input_len], out), 1).cpu().detach().numpy())
        else:
            np.save('./seq/truth_seq_'+args.dataset+'.npy',load_data.cpu().detach().numpy())
            if is_attack:
                if args.dataset == "H36M":
                    pert_save = recoverh36m_3d.recoverh36m_3d(load_data[:, :args.input_len], pert_input[:, :args.input_len])
                elif args.dataset == "CMU":
                    pert_save = recoverCMU_3d.recoverCMU_3d(load_data[:, :args.input_len], pert_input[:, :args.input_len]) 
                elif args.dataset == "3DPW":
                    pert_save = pert_input[:, :args.input_len]
                np.save('./seq/pert_seq_'+args.dataset+'.npy',
                        torch.cat((pert_save, out), 1).cpu().detach().numpy())
            else:
                np.save('./seq/clean_seq_'+args.dataset+'.npy',
                        torch.cat((load_data[:, :args.input_len], out), 1).cpu().detach().numpy())
        #Calculate the errors of bone length, velocity and acceleration bet
        if is_attack:
            clean_p3d = input_seq.clone()
            pert_p3d = pert_input[:, :args.input_len,: ,:].clone()
            BL_clean = Attack.Get_Bone_Length(clean_p3d, connectivity_dict)
            BL_pert = Attack.Get_Bone_Length(pert_p3d, connectivity_dict)
            total_LBL = total_LBL + torch.norm(BL_pert-BL_clean, p=1)
            for m in range(2):
                der_clean = Attack.Derivative_Spatial(clean_p3d)
                der_pert = Attack.Derivative_Spatial(pert_p3d)
                v_dif = der_pert - der_clean
                if m == 0:
                    total_vel += torch.norm(v_dif, p=1)
                elif m == 1:
                    total_acc += torch.norm(v_dif, p=1)
                clean_p3d = der_clean
                pert_p3d = der_pert
        #Calculate MPJPE
        for i in range(args.seq_len - args.input_len):
            mpjpe1[0][i] += (
                        torch.mean(
                            torch.norm(
                                out[:, i, :, :].contiguous().view(-1, 3)
                                - gth[:, i, :, :].contiguous().view(-1, 3),
                                2,
                                1,
                            )
                        ).item()
                        * n
                    )
        N += n
        bar.suffix = "batch time {:.4f}s|total time{:.2f}s".format(
            time.time() - batch_time, time.time() - start_time)
        bar.next()
    bar.finish()
    if is_attack:
        return mpjpe1[0]/N, total_LBL/(N*args.input_len*len(connectivity_dict)), total_vel/(N*args.input_len*args.joints_input), total_acc/(N*args.input_len*args.joints_input)
    else:
        return mpjpe1[0]/N, 0, 0 ,0

if __name__ == '__main__':
    args = parser.parse_args()
    device = args.device
    if device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES']= '0'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        device_ids = [0]
    main()


