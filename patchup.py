import os, sys
from enum import Enum

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
def to_one_hot(inp, num_classes):
    """
    creates a one hot encoding that is a representation of categorical variables as binary vectors for the given label.
    Args:
        inp: label of a sample.
        num_classes: the number of labels or classes that we have in the multi class classification task.
    Returns:
        one hot encoding vector of the specific target.
    """
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)

    return Variable(y_onehot.to(device), requires_grad=False)


class PatchUpMode(Enum):
    SOFT = 'soft'
    HARD = 'hard'


class PatchUp(nn.Module):
    """
    PatchUp Module.
    This module is responsible for applying either Soft PatchUp or Hard PatchUp after a Convolutional module
    or convolutional residual block.
    """
    def __init__(self, block_size=3, gamma=0.9, patchup_type=PatchUpMode.HARD, num_classes=1211):
        """
        PatchUp constructor.
        Args:
            block_size: An odd integer number that defines the size of blocks in the Mask that defines
            the continuous feature should be altered.
            gamma: It is float number in [0, 1]. The gamma in PatchUp decides the probability of altering a feature.
            patchup_type: It is an enum type of PatchUpMode. It defines PatchUp type that can be either
            Soft PatchUp or Hard PatchUp.
        """
        super(PatchUp, self).__init__()
        self.patchup_type = patchup_type
        self.block_size = block_size
        self.gamma = gamma
        self.gamma_adj = None
        self.kernel_size = block_size
        self.stride = 1
        self.padding = block_size // 2
        self.computed_lam = None
        self.num_classes = num_classes

    def adjust_gamma(self, x):
        """
        The gamma in PatchUp decides the probability of altering a feature.
        This function is responsible to adjust the probability based on the
        gamma value since we are altering a continues blocks in feature maps.
        Args:
            x: feature maps for a minibatch generated by a convolutional module or convolutional layer.
        Returns:
            the gamma which is a float number in [0, 1]
        """
        return self.gamma * x.shape[-1] ** 2 / \
               (self.block_size ** 2 * (x.shape[-1] - self.block_size + 1) ** 2)

    def forward(self, x, targets=None, lam=None, patchup_type=PatchUpMode.SOFT):
        """
            Forward pass in for the PatchUp Module.
        Args:
            x: Feature maps for a mini-batch generated by a convolutional module or convolutional layer.
            targets: target of samples in the mini-batch.
            lam: In a case, you want to apply PatchUp for a fixed lambda instead of sampling from the Beta distribution.
            patchup_type: either Hard PatchUp or Soft PatchUp.
        Returns:
            the interpolated hidden representation using PatchUp.
            target_a: targets associated with the first samples in the randomly selected sample pairs in the mini-batch.
            target_b: targets associated with the second samples in the randomly selected sample pairs in the mini-batch.
            target_reweighted: target target re-weighted for interpolated samples after altering patches
            with either Hard PatchUp or Soft PatchUp.
            x: interpolated hidden representations.
            total_unchanged_portion: the portion of hidden representation that remained unchanged after applying PatchUp.
        """
        # print('targets', targets)
        self.patchup_type = patchup_type
        if type(self.training) == type(None):
            Exception("model's mode is not set in to neither training nor testing mode")
        if not self.training:
            # if the model is at the inference time (evaluation or test), we are not applying patchUp.
            return x, targets

        if type(lam) == type(None):
            # if we are not using fixed lambda, we should sample from the Beta distribution with fixed alpha equal to 2.
            # lambda will be a float number in range [0, 1].
            lam = np.random.beta(2.0, 2.0)

        if self.gamma_adj is None:
            self.gamma_adj = self.adjust_gamma(x)
        p = torch.ones_like(x[0]) * self.gamma_adj
        # For each feature in the feature map, we will sample from Bernoulli(p). If the result of this sampling
        # for feature f_{ij} is 0, then Mask_{ij} = 1. If the result of this sampling for f_{ij} is 1,
        # then the entire square region in the mask with the center Mask_{ij} and the width and height of
        # the square of block_size is set to 0.
        m_i_j = torch.bernoulli(p)
        mask_shape = len(m_i_j.shape)

        # after creating the binary Mask. we are creating the binary Mask created for first sample as a pattern
        # for all samples in the minibatch as the PatchUp binary Mask. to do so, we can just expnand the pattern
        # created for the first sample.
        m_i_j = m_i_j.expand(x.size(0), m_i_j.size(0), m_i_j.size(1)).to(device)
        
        # following line provides the continues blocks that should be altered with PatchUp denoted as holes here.
        holes = F.max_pool1d(m_i_j, self.kernel_size, self.stride, self.padding)
        # following line gives the binary mask that contains 1 for the features that should be remain unchanged and 1
        # for the features that lie in the continues blocks that selected for interpolation.
        mask = 1 - holes
        unchanged = mask * x
        print(torch.all(mask[0, 0].eq(mask[0, 1])), mask.shape)
        if mask_shape == 1:
            total_feats = x.size(1)
        else:
            total_feats = x.size(1) * x.size(2)
        total_changed_pixels = holes[0].sum()
        total_changed_portion = total_changed_pixels / total_feats
        total_unchanged_portion = (total_feats - total_changed_pixels) / total_feats
        # following line gives the indices of second ssamples in the pair permuted randomly.
        indices = np.random.permutation(x.size(0))
        targets = to_one_hot(targets, self.num_classes)
        target_shuffled_onehot = targets[indices]
        patches = None
        target_reweighted = None
        target_b = None
        if self.patchup_type == PatchUpMode.SOFT:
            # apply Soft PatchUp combining operation for the selected continues blocks.
            target_reweighted = total_unchanged_portion * targets +  lam * total_changed_portion * targets + \
                                target_shuffled_onehot * (1 - lam) * total_changed_portion
            patches = holes * x
            patches = patches * lam + patches[indices] * (1 - lam)
            target_b = lam * targets + (1 - lam) * target_shuffled_onehot
        elif self.patchup_type == PatchUpMode.HARD:
            # apply Hard PatchUp combining operation for the selected continues blocks.
            target_reweighted = total_unchanged_portion * targets + total_changed_portion * target_shuffled_onehot
            patches = holes * x
            patches = patches[indices]
            target_b = targets[indices]
        x = unchanged + patches
        target_a = targets
        return target_a, target_b, target_reweighted, x, total_unchanged_portion




 [1] "ENSG00000000003" "ENSG00000000005" "ENSG00000000419" "ENSG00000000457"
 [5] "ENSG00000000460" "ENSG00000000938" "ENSG00000000971" "ENSG00000001036" 
 [9] "ENSG00000001084" "ENSG00000001167" "ENSG00000001460" "ENSG00000001461" 
[13] "ENSG00000001497" "ENSG00000001561" "ENSG00000001617" "ENSG00000001626" 
[17] "ENSG00000001629" "ENSG00000002016" "ENSG00000002079" "ENSG00000002330" 
[21] "ENSG00000002549" "ENSG00000002586" "ENSG00000002587" "ENSG00000002726" 
[25] "ENSG00000002745"