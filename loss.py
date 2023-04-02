'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
'''

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from tools import *

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Phoneme_SSL_loss(nn.Module):
    def __init__(self, num_frames, num_sample=5):
        super(Phoneme_SSL_loss, self).__init__()
        self.all_ind = torch.LongTensor(list(range(num_frames)))
        self.num_sample = num_sample

    def get_output_phn(self, outputs, seq_len):
        output_ = 0
        for i in range(len(seq_len.tolist())):
            length = seq_len[i]
            output = outputs[i, :length, :, :]
            if i == 0:
                output_ = output
            else:
                output_ = torch.cat((output_, output), dim=0)
        return output_

    def forward(self, output, seq_len):
        output_seg = self.get_output_phn(output, seq_len)
        num_seg, num_frame, dim = output_seg.size()
        sim_pos = F.cosine_similarity(output_seg[:, :-1, :], output_seg[:, 1:, :], dim=-1).unsqueeze(-1)
        output_seg_group = output_seg.transpose(0, 1).reshape(num_frame, -1)
        random_index = [torch.randperm(num_frame - 3).tolist()[:self.num_sample] for i in range(num_frame-1)]
        output_frames = [
            output_seg_group[(self.all_ind != i - 1) * (self.all_ind != i) * (self.all_ind != i + 1), :][random_index[i],
            :] for i in self.all_ind[:-1]]
        negatives = torch.cat(output_frames, dim=0).reshape(-1, num_seg, dim).transpose(0, 1)
        anchors = output_seg[:, :-1, :].repeat(1, 1, self.num_sample).reshape(num_seg, -1, dim)
        sim_neg = F.cosine_similarity(anchors, negatives, dim=-1).reshape(-1, 19, self.num_sample)
        sim_all = torch.cat((sim_pos, sim_neg), dim=-1)
        loss_seg = torch.mean(torch.mean(-F.log_softmax(sim_all, dim=-1)[:,:, 0], dim=-1))
        return 
        
class AAMsoftmax(nn.Module):
    def __init__(self, n_class, m, s):
        
        super(AAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 192), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        
        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return loss, prec1