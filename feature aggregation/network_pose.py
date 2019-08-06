import copy
import math
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, Bottleneck
import torch.nn.init as init
import numpy as np
import cv2
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.structures.bounding_box import BoxList
num_classes = 107  # change this depend on your dataset

def cal_lean_bbox(x1, y1, x2, y2):
    x_len = math.fabs(x1 - x2)
    y_len = math.fabs(y1 - y2)
    z_len = math.sqrt(x_len * x_len + y_len * y_len)
    rec_y = z_len * 0.5
    rec_x = z_len * 0.3
    if x1 < x2:
        new_x1, new_y1 = int(x1 - (rec_x / z_len) * y_len), int(y1 + (rec_x / z_len) * x_len)
        new_x2, new_y2 = int(x1 + (rec_x / z_len) * y_len), int(y1 - (rec_x / z_len) * x_len)
        new_x3, new_y3 = int(x2 + (rec_x / z_len) * y_len), int(y2 - (rec_x / z_len) * x_len)
        new_x4, new_y4 = int(x2 - (rec_x / z_len) * y_len), int(y2 + (rec_x / z_len) * x_len)
    else:
        new_x1, new_y1 = int(x1 - (rec_x / z_len) * y_len), int(y1 - (rec_x / z_len) * x_len)
        new_x2, new_y2 = int(x1 + (rec_x / z_len) * y_len), int(y1 + (rec_x / z_len) * x_len)
        new_x3, new_y3 = int(x2 + (rec_x / z_len) * y_len), int(y2 + (rec_x / z_len) * x_len)
        new_x4, new_y4 = int(x2 - (rec_x / z_len) * y_len), int(y2 - (rec_x / z_len) * x_len)

    return [(np.clip(new_x1, 0, 286), np.clip(new_y1, 0, 126)),
            (np.clip(new_x2, 0, 286), np.clip(new_y2, 0, 126)),
            (np.clip(new_x3, 0, 286), np.clip(new_y3, 0, 126)),
            (np.clip(new_x4, 0, 286), np.clip(new_y4, 0, 126))]

def part_bbox(point1, point2):
    if point1[2] == 0 or point2[2] == 0:
        return [0] * 4 , 0

    else:
        list_lean_points = cal_lean_bbox(point1[0], point1[1], point2[0],point2[1])
        x, y, w, h = cv2.boundingRect(np.array(list_lean_points))

        return [x, y,x+w,y+h] , 1

def generate_part(kpts):
    cnt_body ,      flag_body = part_bbox(kpts[2],kpts[13])
    cnt_leftfoot,   flag_leftfoot = part_bbox(kpts[5], kpts[6])
    cnt_rightfoot,  flag_rightfoot = part_bbox(kpts[3], kpts[4])
    cnt_left_hum,   flag_left_hum = part_bbox(kpts[10], kpts[11])
    cnt_left_shank, flag_left_shank = part_bbox(kpts[11], kpts[12])
    cnt_right_hum,  flag_right_hum = part_bbox(kpts[7], kpts[8])
    cnt_right_shank,flag_right_shank = part_bbox(kpts[8], kpts[9])
    proposal = [cnt_body, cnt_leftfoot, cnt_rightfoot, cnt_left_hum, cnt_right_hum, cnt_left_shank, cnt_right_shank]
    visual   = [flag_body,flag_leftfoot,flag_rightfoot,flag_left_hum,flag_right_hum,flag_left_shank,flag_right_shank]
    return proposal , visual

def generate_proposal(kpts):
    proposls = []
    visuables = []
    for kpt_img_per in kpts:
        proposl,visuable = generate_part(kpt_img_per)
        proposls.append(proposl)
        visuables.append(visuable)
    return proposls,visuables


class p_roiAlign(nn.Module):
    def __init__(self,reslution):
        super(p_roiAlign, self).__init__()
        pooler = Pooler(
            output_size=(reslution, reslution),
            scales=(0.0625,),
            sampling_ratio=2
        )
        self.pooler = pooler

    def forward(self, x,proposal):
        x = self.pooler(x,proposal)

        return x

class MGN(nn.Module):
    def __init__(self):
        super(MGN, self).__init__()

        feats = 256
        resnet = resnet50(pretrained=True)

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

        self.p4 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(4, 9))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(8, 18))
        self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(8, 18))
        self.maxpool_zg_p4 = nn.MaxPool2d(kernel_size=(8, 18))
        self.maxpool_zp2 = nn.MaxPool2d(kernel_size=(8, 9))
        self.maxpool_zp3 = nn.MaxPool2d(kernel_size=(8, 6))

        self.maxpool_zpbody = nn.MaxPool2d(kernel_size=(4, 4))
        self.maxpool_zppart = nn.MaxPool2d(kernel_size=(2, 2))

        self.reduction = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())

        self.reduction_body = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())
        self.reduction_part = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())


        self._init_reduction(self.reduction)
        self._init_reduction(self.reduction_body)
        self._init_reduction(self.reduction_part)


        self.fc_id_2048_0 = nn.Linear(feats, num_classes)
        self.fc_id_2048_1 = nn.Linear(feats, num_classes)
        self.fc_id_2048_2 = nn.Linear(feats, num_classes)
        self.fc_id_2048_3 = nn.Linear(feats, num_classes)

        self.fc_id_256_1_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_1_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_2 = nn.Linear(feats, num_classes)


        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        self._init_fc(self.fc_id_2048_3)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

        # self._init_fc(self.fc_id_256_3_0)
        # self._init_fc(self.fc_id_256_3_1)
        # self._init_fc(self.fc_id_256_3_2)
        # self._init_fc(self.fc_id_256_3_3)
        # self._init_fc(self.fc_id_256_3_4)
        # self._init_fc(self.fc_id_256_3_5)
        # self._init_fc(self.fc_id_256_3_6)

        self.pool_body = p_roiAlign(4)
        self.pool_part = p_roiAlign(2)


    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x, kpts):
        x = self.backbone(x)

        proposal , visuable = generate_proposal(kpts)
        proposal_body = [BoxList([propals_per_img[0]], (288, 128)).to('cuda') for propals_per_img in proposal]
        proposal_part = [BoxList(propals_per_img[1:], (288, 128)).to('cuda') for propals_per_img in proposal]

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        p_pool_body = self.pool_body([p4], proposal_body)
        p_pool_part = self.pool_part([p4], proposal_part)

        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)
        zg_p4 = self.maxpool_zg_p4(p4)

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, :, 0:1]
        z1_p2 = zp2[:, :, :, 1:2]

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, :, 0:1]
        z1_p3 = zp3[:, :, :, 1:2]
        z2_p3 = zp3[:, :, :, 2:3]


        body_p4 = self.maxpool_zpbody(p_pool_body)
        bacth_per_img = body_p4.shape[0]
        part0_p4 = self.maxpool_zppart(p_pool_part[0:bacth_per_img, :, :, :])
        part1_p4 = self.maxpool_zppart(p_pool_part[bacth_per_img:bacth_per_img*2, :, :, :])
        part2_p4 = self.maxpool_zppart(p_pool_part[bacth_per_img*2:bacth_per_img*3, :, :, :])
        part3_p4 = self.maxpool_zppart(p_pool_part[bacth_per_img*3:bacth_per_img*4, :, :, :])
        part4_p4 = self.maxpool_zppart(p_pool_part[bacth_per_img*4:bacth_per_img*5, :, :, :])
        part5_p4 = self.maxpool_zppart(p_pool_part[bacth_per_img*5:bacth_per_img*6, :, :, :])

        fg_p1 = self.reduction(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction(zg_p3).squeeze(dim=3).squeeze(dim=2)

        fg_p4 = self.reduction(zg_p4).squeeze(dim=3).squeeze(dim=2)

        f0_p2 = self.reduction(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction(z2_p3).squeeze(dim=3).squeeze(dim=2)

        f_body_p4 = self.reduction_body(body_p4).squeeze(dim=3).squeeze(dim=2)
        f_part0_p4 = self.reduction_part(part0_p4).squeeze(dim=3).squeeze(dim=2)
        f_part1_p4 = self.reduction_part(part1_p4).squeeze(dim=3).squeeze(dim=2)
        f_part2_p4 = self.reduction_part(part2_p4).squeeze(dim=3).squeeze(dim=2)
        f_part3_p4 = self.reduction_part(part3_p4).squeeze(dim=3).squeeze(dim=2)
        f_part4_p4 = self.reduction_part(part4_p4).squeeze(dim=3).squeeze(dim=2)
        f_part5_p4 = self.reduction_part(part5_p4).squeeze(dim=3).squeeze(dim=2)

        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)
        l_p3 = self.fc_id_2048_2(fg_p3)

        l_p4 = self.fc_id_2048_3(fg_p4)

        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        l2_p3 = self.fc_id_256_2_2(f2_p3)

        list_for_pose_loss = [f_body_p4, f_part0_p4,  f_part1_p4, f_part2_p4, f_part3_p4, f_part4_p4, f_part5_p4, visuable]

        predict = torch.cat([fg_p1, fg_p2, fg_p3, fg_p4, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)

        return predict, fg_p1, fg_p2, fg_p3, fg_p4, l_p1, l_p2, l_p3, l_p4, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3, list_for_pose_loss
