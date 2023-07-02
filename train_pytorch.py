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

def main():
    assert args.dataset in ["H36M", "CMU", "3DPW"]
    print(">>> loading data")
    input_n = args.input_len
    output_n = args.seq_len - args.input_len
    # load datasets for training
    if args.dataset=="H36M":
        acts = data_utils.define_actions(args.actions)
        train_dataset = H36motion3D(
            path_to_data=args.data_dir,
            actions=args.actions,
            input_n=input_n,
            output_n=output_n,
            split=0
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
    elif args.dataset=="3DPW":
        train_dataset = Pose3dPW3D(
            path_to_data=args.data_dir_3dpw,
            input_n=input_n,
            output_n=output_n,
            split=0,
        )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.job,
        pin_memory=True,
    )
    # load datasets for testing
    test_data = dict()
    if args.dataset=="H36M":
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
    print(">>> train data {}".format(train_dataset.__len__()))
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
    if args.is_load == True:
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
    epochs = args.epochs
    #Train & test
    for epoch in range(epoch_pretrained, epochs):
        
        ret_log = np.array([epoch + 1])
        head = np.array(["epoch"])

        l_t = train(model, optimizer, scheduler, train_loader, epoch)
        head = np.append(head, ["loss_train"])
        if args.dataset == "3DPW":
            ret_log = np.append(ret_log, [l_t*1000])

            test_3d = test(model, test_loader)
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
            ret_log = np.append(ret_log, [l_t])
            for act in acts:
                test_3d = test(model, test_data[act])
                ret_log = np.append(ret_log, test_3d.cpu().numpy())
                head = np.append(
                    head, [act + "3d80", act + "3d160", act + "3d320", act + "3d400"]
                )
                if output_n > 10:
                    head = np.append(head, [act + "3d560", act + "3d1000"])

        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
        if epoch == 0:
            df.to_csv("./"+args.dataset+"_TrajCNN_Rec.csv", header=head, index=False)
        else:
            with open("./"+args.dataset+"_TrajCNN_Rec.csv", "a") as f:
                df.to_csv(f, header=False, index=False)

        save_state = {'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch_pretrained': epoch+1}
        if not os.path.exists(os.path.join(args.checkpoint, args.dataset)):
            os.makedirs(os.path.join(args.checkpoint, args.dataset))
        torch.save(save_state,
                    os.path.join(args.checkpoint, args.dataset, args.dataset+'_TrajectoryCNN_'+str(epoch)+'.pth'))
        print('model saving done!')

def train(model, optimizer, scheduler, train_loader, epoch):
    start_time = time.time()
    model.train()
    bar = Bar(">>>", fill=">", max=len(train_loader))
    for batch_id, (input_seq, output_seq, load_data) in enumerate(train_loader):
        batch_time = time.time()
        load_data = load_data.float().to(device)
        input_seq = input_seq.float().to(device)
        output_seq = output_seq.float().to(device)

        #Train with the normal sequence
        load_data = load_data[:, :, 0:args.joints_input, :]
        input_seq = input_seq[:, :, :args.joints_input, :]
        last_frame = input_seq[:, args.input_len - 1, :]
        last_frame = last_frame.unsqueeze(1)
        last_frame_repeat = last_frame.repeat(1, args.seq_len - args.input_len, 1, 1)
        input_data = torch.cat((input_seq, last_frame_repeat), 1)
        gth = output_seq[:, :, 0:args.joints_input, :]
        input_data = input_data.to(device)
        gth = gth.to(device)
        out, p = model(input_data)
        
        loss = 0
        loss_input = out - gth
        loss = torch.mean(torch.norm(loss_input, p=2, dim=3, keepdim=True))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Train with the reverse sequence
        load_data_r = torch.flip(load_data, dims = [1])
        input_seq_r = load_data_r[:, :args.input_len, :, :]
        last_frame_r = load_data_r[:, args.input_len - 1, :]
        last_frame_r = last_frame_r.unsqueeze(1)
        last_frame_r_repeat = last_frame_r.repeat(1, args.seq_len - args.input_len, 1, 1)
        input_data_r = torch.cat((input_seq_r, last_frame_r_repeat), 1)
        gth_r = load_data_r[:, args.input_len: args.seq_len, :, :]
        input_data_r = input_data_r.to(device)
        gth_r = gth_r.to(device)
        out_r, p_r = model(input_data_r)

        loss_r = 0
        loss_input_r = out_r - gth_r
        loss_r = torch.mean(torch.norm(loss_input_r, p=2, dim=3, keepdim=True))

        optimizer.zero_grad()
        loss_r.backward()
        optimizer.step()

        bar.suffix = "{}/{}|epoch {}|batch time {:.4f}s|total time{:.2f}s|loss: {}".format(
            batch_id + 1, len(train_loader), epoch, time.time() - batch_time, time.time() - start_time, (loss.item()+loss_r.item())/2
            )
        bar.next()
    bar.finish()
    scheduler.step()

    return (loss.item()+loss_r.item())/2


def test(model, test_loader):
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

    mpjpe_x = evaluate(model, test_loader)
    mpjpe += mpjpe_x

    return mpjpe[0][eval_frame]


def evaluate(model, test_loader):
    model.eval()
    N = 0
    mpjpe1 = torch.zeros((1, args.seq_len - args.input_len)).to(device)

    for batch_id, (input_seq, output_seq, load_data) in enumerate(test_loader):
        n = load_data.shape[0]
        load_data = load_data.float().to(device)
        input_seq = input_seq.float().to(device)
        output_seq = output_seq.float().to(device)

        gth = load_data[:, args.input_len:, :, :]
        load_data = load_data[:, :, :args.joints_input, :]
        input_seq = input_seq[:, :, :args.joints_input, :]
        last_frame = input_seq[:, args.input_len - 1, :]
        last_frame = last_frame.unsqueeze(1)
        last_frame_repeat = last_frame.repeat(1, args.seq_len - args.input_len, 1, 1)
        input_data = torch.cat((input_seq, last_frame_repeat), 1)
        input_data = input_data.to(device)
        out, p = model(input_data)
        #Recover the sequence to the original dimension
        if args.dataset == "H36M":
            out = recoverh36m_3d.recoverh36m_3d(gth, out)
        elif args.dataset == "CMU":
            out = recoverCMU_3d.recoverCMU_3d(gth, out)
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

    return mpjpe1[0]/N


if __name__ == '__main__':
    args = parser.parse_args()
    device = args.device
    if device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES']= '0'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        device_ids = [0]
    main()


