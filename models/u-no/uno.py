# Codes for section: Results on Darcy Flow Equation

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import matplotlib.pyplot as plt
from integral_operators import *
import operator
from functools import reduce
from functools import partial

from timeit import default_timer

###############
#  UNO^dagger achitechtures
###############
class UNO1d(nn.Module):
    def __init__(self, num_channels, width, pad = 6, factor = 1, initial_step = 10):
        super(UNO1d, self).__init__()

        self.in_width = num_channels * initial_step + 1 # input channel
        self.width = width 
        
        self.padding = pad  # pad the domain if input is non-periodic

        self.fc_n1 = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = OperatorBlock_1D(self.width, 2*factor*self.width,20, 9)

        self.conv1 = OperatorBlock_1D(2*factor*self.width, 4*factor*self.width, 10, 4, Normalize = True)

        self.conv2 = OperatorBlock_1D(4*factor*self.width, 4*factor*self.width, 10, 4)

        self.conv4 = OperatorBlock_1D(4*factor*self.width, 2*factor*self.width, 20, 4, Normalize = True)

        self.conv5 = OperatorBlock_1D(4*factor*self.width, self.width, 43, 9) # will be reshaped

        self.fc1 = nn.Linear(2*self.width, 1*self.width)
        self.fc2 = nn.Linear(1*self.width, num_channels)

    def forward(self, x, grid):
        x = torch.cat((x, grid), dim=-1)

        x_fc_1 = self.fc_n1(x)
        x_fc_1 = F.gelu(x_fc_1)

        x_fc0 = self.fc0(x_fc_1)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 2, 1)
        # scale = math.ceil(x_fc0.shape[-1]/43)
        x_fc0 = F.pad(x_fc0, [0,self.padding])
        
        D1 = x_fc0.shape[-1]
        x_c0 = self.conv0(x_fc0,D1//2)

        x_c1 = self.conv1(x_c0,D1//4)

        x_c2 = self.conv2(x_c1,D1//4)
        
        x_c4 = self.conv4(x_c2 ,D1//2)
        x_c4 = torch.cat([x_c4, x_c0], dim=1)

        x_c5 = self.conv5(x_c4,D1)
        x_c5 = torch.cat([x_c5, x_fc0], dim=1)


        if self.padding!=0:
            x_c5 = x_c5[..., :-self.padding]


        x_c5 = x_c5.permute(0, 2, 1)

        x_fc1 = self.fc1(x_c5)
        x_fc1 = F.gelu(x_fc1)

        x_out = self.fc2(x_fc1)
        return x_out.unsqueeze(-2)

class UNO2d(nn.Module):
    def __init__(self, num_channels, width, pad = 6, factor = 1, initial_step = 10):
        super(UNO2d, self).__init__()

        self.in_width = num_channels * initial_step + 2 # input channel
        self.width = width 
        
        self.padding = pad  # pad the domain if input is non-periodic

        self.fc_n1 = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = OperatorBlock_2D(self.width, 2*factor*self.width,20, 20, 9, 9)

        self.conv1 = OperatorBlock_2D(2*factor*self.width, 4*factor*self.width, 10, 10, 4,4, Normalize = True)

        self.conv2 = OperatorBlock_2D(4*factor*self.width, 4*factor*self.width, 10, 10,4,4)

        self.conv4 = OperatorBlock_2D(4*factor*self.width, 2*factor*self.width, 20, 20,4,4, Normalize = True)

        self.conv5 = OperatorBlock_2D(4*factor*self.width, self.width, 43, 43,9,9) # will be reshaped

        self.fc1 = nn.Linear(2*self.width, 1*self.width)
        self.fc2 = nn.Linear(1*self.width, num_channels)

    def forward(self, x, grid):
        x = torch.cat((x, grid), dim=-1)

        x_fc_1 = self.fc_n1(x)
        x_fc_1 = F.gelu(x_fc_1)

        x_fc0 = self.fc0(x_fc_1)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 3, 1, 2)
        
        # scale = math.ceil(x_fc0.shape[-1]/85)
        x_fc0 = F.pad(x_fc0, [0,self.padding, 0,self.padding])
        
        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]

        x_c0 = self.conv0(x_fc0,D1//2,D2//2)

        x_c1 = self.conv1(x_c0,D1//4,D2//4)

        x_c2 = self.conv2(x_c1,D1//4,D2//4)
        
        x_c4 = self.conv4(x_c2 ,D1//2,D2//2)
        x_c4 = torch.cat([x_c4, x_c0], dim=1)

        x_c5 = self.conv5(x_c4,D1,D2)
        x_c5 = torch.cat([x_c5, x_fc0], dim=1)


        if self.padding!=0:
            x_c5 = x_c5[..., :-self.padding, :-self.padding]


        x_c5 = x_c5.permute(0, 2, 3, 1)

        x_fc1 = self.fc1(x_c5)
        x_fc1 = F.gelu(x_fc1)

        x_out = self.fc2(x_fc1)
        
        return x_out.unsqueeze(-2)
    