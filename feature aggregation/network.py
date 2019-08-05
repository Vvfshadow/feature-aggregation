import copy
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, Bottleneck
import torch.nn.init as init
import numpy as np
num_classes = 107  # change this depend on your dataset

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim // 2

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.chanel_in, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.chanel_in, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.query_conv = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=self.chanel_in, kernel_size=1),
        #                                 nn.BatchNorm2d(self.chanel_in),
                                        # nn.Tanh())
        # self.key_conv = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=self.chanel_in, kernel_size=1),
        #                               nn.BatchNorm2d(self.chanel_in),
                                      # nn.Tanh())
        # self.value_conv = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1),
        #                                 nn.BatchNorm2d(in_dim),
                                        # nn.Tanh())
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        energy = energy / np.sqrt(C)
        attention = self.softmax(energy)
        N = energy.size(-1)
        attention = attention / N
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        # out = out + x
        out = self.tanh(out)
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy = energy / np.sqrt(height*width)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)

        N = energy.size(-1)
        attention = attention / N
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        # out = self.tanh(out)
        return out


class DAHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DAHead, self).__init__()
        inter_channels = in_channels

        # self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #                             nn.BatchNorm2d(inter_channels),
        #                             nn.ReLU())
        #
        # self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #                             nn.BatchNorm2d(inter_channels),
        #                             nn.ReLU())
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sa = PAM_Module(inter_channels)
        # self.sa = GC_Module(in_channels, inter_channels, 'att', ['channel_add', 'channel_mul'])
        for m in self.sa.modules():
            if isinstance(m, nn.Conv2d):
                # init.xavier_uniform(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    init.constant(m.bias, 0)

        self.sc = CAM_Module(inter_channels)
        for m in self.sc.modules():
            if isinstance(m, nn.Conv2d):
                # init.xavier_uniform(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    init.constant(m.bias, 0)

        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        for m in self.conv51.modules():
            if isinstance(m, nn.Conv2d):
                # init.xavier_uniform(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        for m in self.conv52.modules():
            if isinstance(m, nn.Conv2d):
                # init.xavier_uniform(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        for m in self.conv6.modules():
            if isinstance(m, nn.Conv2d):
                # init.xavier_uniform(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        for m in self.conv7.modules():
            if isinstance(m, nn.Conv2d):
                # init.xavier_uniform(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(out_channels, out_channels, 1))
        for m in self.conv8.modules():
            if isinstance(m, nn.Conv2d):
                # init.xavier_uniform(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        # feat1 = self.conv5a(x)
        sa_feat = self.sa(x)
        # sa_conv = self.conv51(sa_feat)
        # sa_output = self.conv6(sa_conv)

        # feat2 = self.conv5c(x)
        sc_feat = self.sc(x)
        # sc_conv = self.conv52(sc_feat)
        # sc_output = self.conv7(sc_conv)

        feat_sum = sa_feat + self.gamma * sc_feat
        sasc_output = self.conv8(feat_sum)
        return sasc_output


class MGN(nn.Module):
    def __init__(self):
        super(MGN, self).__init__()

        feats = 256
        resnet = resnet50(pretrained=True)

        self.attention1 = DAHead(2048, 2048)
        self.attention2 = DAHead(2048, 2048)
        self.attention3 = DAHead(2048, 2048)


        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(4, 9))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(8, 18))
        self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(8, 18))
        self.maxpool_zp2 = nn.MaxPool2d(kernel_size=(8, 9))
        self.maxpool_zp3 = nn.MaxPool2d(kernel_size=(8, 6))

        self.reduction = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())

        self._init_reduction(self.reduction)

        self.fc_id_2048_0 = nn.Linear(feats, num_classes)
        self.fc_id_2048_1 = nn.Linear(feats, num_classes)
        self.fc_id_2048_2 = nn.Linear(feats, num_classes)

        self.fc_id_256_1_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_1_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_2 = nn.Linear(feats, num_classes)
        #********************************
        # self.fc_id_256_2_3 = nn.Linear(feats, num_classes)
        # self.fc_id_256_2_4 = nn.Linear(feats, num_classes)
        # self.fc_id_256_2_5 = nn.Linear(feats, num_classes)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)
        #********************************
        # self._init_fc(self.fc_id_256_2_3)
        # self._init_fc(self.fc_id_256_2_4)
        # self._init_fc(self.fc_id_256_2_5)


    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):
        x = self.backbone(x)


        # p1 = self.attention1(self.p1(x))
        # p2 = self.attention2(self.p2(x))
        # p3 = self.attention3(self.p3(x))
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, :, 0:1]
        z1_p2 = zp2[:, :, :, 1:2]

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, :, 0:1]
        z1_p3 = zp3[:, :, :, 1:2]
        z2_p3 = zp3[:, :, :, 2:3]
        # ********************************
        # z3_p3 = zp3[:, :, :, 3:4]
        # z4_p3 = zp3[:, :, :, 4:5]
        # z5_p3 = zp3[:, :, :, 5:6]



        fg_p1 = self.reduction(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction(z2_p3).squeeze(dim=3).squeeze(dim=2)
        # *************************************************
        # f3_p3 = self.reduction(z3_p3).squeeze(dim=3).squeeze(dim=2)
        # f4_p3 = self.reduction(z4_p3).squeeze(dim=3).squeeze(dim=2)
        # f5_p3 = self.reduction(z5_p3).squeeze(dim=3).squeeze(dim=2)


        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)
        l_p3 = self.fc_id_2048_2(fg_p3)

        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        l2_p3 = self.fc_id_256_2_2(f2_p3)
        #*****************************************************
        # l3_p3 = self.fc_id_256_2_3(f3_p3)
        # l4_p3 = self.fc_id_256_2_4(f4_p3)
        # l5_p3 = self.fc_id_256_2_5(f5_p3)

        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)

        return predict, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3
