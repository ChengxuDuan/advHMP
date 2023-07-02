import torch.nn as nn
import torch


class TrajectoryNet(nn.Module):
    def __init__(self, input_n=10, output_n=10, dropout=0.25):
        super(TrajectoryNet, self).__init__()
        
        self.TB_block0 = Trajectory_Block(dropout)
        self.TB_block1 = Trajectory_Block(dropout)
        self.TB_block2 = Trajectory_Block(dropout)
        self.TB_block3 = Trajectory_Block(dropout)

        self.layer_0 = torch.nn.Sequential(
            nn.Conv2d(input_n+output_n, 64, 1, stride=1, padding=0),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)

        )
        self.layer_1 = torch.nn.Sequential(
            nn.Conv2d(64, output_n, 3, stride=1, padding=1),
            # nn.BatchNorm2d(output_n),
            nn.LeakyReLU(negative_slope=0.2)

        )
        self.layer_2 = torch.nn.Sequential(
            nn.Conv2d(output_n, output_n, 1, stride=1, padding=0),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('leaky_relu'))
                # nn.init.kaiming_uniform_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.layer_0(x)
        # print('aaaaaaaaaaaaaaaaaa')
        p = x.data.cpu().numpy().copy()
        x = self.TB_block0(x)
        x = self.TB_block1(x)
        x = self.TB_block2(x)
        x = self.TB_block3(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        return x, p


class Trajectory_Block(nn.Module):
    def __init__(self, dropout=0.75):
        super(Trajectory_Block, self).__init__()
        self.TB_foward_0 = torch.nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout)

        )
        self.TB_foward_1 = torch.nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout)

        )
        self.TB_foward_2 = torch.nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout)

        )
        self.TB_foward_3 = torch.nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout)

        )
        self.TB_foward_4 = torch.nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout)

        )
        self.TB_residual_0 = torch.nn.Sequential(
            nn.Conv2d(64, 64, 1, stride=1, padding=0),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)

        )
        self.TB_residual_1 = torch.nn.Sequential(
            nn.Conv2d(64, 64, 1, stride=1, padding=0),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)

        )
        self.TB_residual_2 = torch.nn.Sequential(
            nn.Conv2d(64, 64, 1, stride=1, padding=0),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)

        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('leaky_relu'))
                # nn.init.kaiming_uniform_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
                
    def forward(self, x):
        # batch_size = x.size(0)
        x_0 = self.TB_residual_0(x)
        traj_1 = self.TB_foward_0(x)
        x_1 = self.TB_residual_1(traj_1)
        traj_2 = self.TB_foward_1(traj_1)
        x_2 = self.TB_residual_2(traj_2)
        traj_3 = self.TB_foward_2(traj_2)
        traj_4 = self.TB_foward_3(traj_3 + x_2)
        traj_5 = self.TB_foward_4(traj_4 + x_1)
        out = traj_5 + x_0
        return out
