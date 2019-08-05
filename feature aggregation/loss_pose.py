from torch.nn import CrossEntropyLoss
from torch.nn.modules import loss
from utils.TripletLoss import TripletLoss
import torch

class Loss(loss._Loss):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, outputs, labels):
        cross_entropy_loss = CrossEntropyLoss()
        triplet_loss = TripletLoss(margin=1.2)
        triplet_loss_pose = TripletLoss(margin=0.8)


        Triplet_Loss = [triplet_loss(output, labels) for output in outputs[1:5]]
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        CrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in outputs[5:-2]]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        list_loss_pose = outputs[-1][:7]
        visuable = torch.Tensor(outputs[-1][-1]).t().to('cuda')

        Triplet_Loss_pose = [triplet_loss_pose(_loss[_vis == 1, :], labels[_vis == 1]) for _loss, _vis in zip(list_loss_pose, visuable) if len(labels[_vis == 1]) != 0]
        Triplet_Loss_pose = sum(Triplet_Loss_pose) / len(Triplet_Loss_pose)

        loss_sum = 2 * Triplet_Loss + 4 * CrossEntropy_Loss + Triplet_Loss_pose

        print('\rtotal loss:%.2f  Triplet_Loss:%.2f  CrossEntropy_Loss:%.2f  Triplet_Loss_pose:%.2f' % (
            loss_sum.data.cpu().numpy(),
            Triplet_Loss.data.cpu().numpy(),
            CrossEntropy_Loss.data.cpu().numpy(),
            Triplet_Loss_pose.data.cpu().numpy()),
              end=' ')
        return loss_sum
