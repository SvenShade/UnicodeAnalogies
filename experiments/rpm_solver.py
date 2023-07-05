# INFORMATION ------------------------------------------------------------------------------------------------------- #


# Author:  Steven Spratley
# Date:    01/11/22
# Purpose: Models for solving PMPs in Unicode Analogies.


# IMPORTS ----------------------------------------------------------------------------------------------------------- #


import sys, torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from   basic_model import BasicModel


# CONSTANTS --------------------------------------------------------------------------------------------------------- #


DR_S, DR_F =  .1, .5      #Dropout prob. for spatial and fully-connected layers.
O_HC, O_OC =  64, 64      #Hidden and output channels for original enc.
F_HC, F_OC =  64, 16      #Hidden and output channels for frame enc.
S_HC, S_OC = 128, 64      #Hidden and output channels for sequence enc.
F_PL, S_PL = 5*5, 16      #Pooled sizes for frame and sequence enc. outputs.
F_Z = F_OC*F_PL           #Frame embedding dimensions.
K_D = 7                   #Conv. kernel dimensions.


# CLASSES ----------------------------------------------------------------------------------------------------------- #


#Helper function for spatial dropout.
class perm(nn.Module):
    def __init__(self):
        super(perm, self).__init__()
    def forward(self, x):
        return x.permute(0, 2, 1)
    
class flat(nn.Module):
    def __init__(self):
        super(flat, self).__init__()
    def forward(self, x):
        return x.flatten(1)
    
#Convolutional block class (conv, elu, bnorm, dropout). If 1D block, no downsampling. If 2D, stride==2.
#Implements spatial dropout for both 1D and 2D convolutional layers.
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dim):
        super(ConvBlock, self).__init__()
        self.conv  = getattr(nn, 'Conv{}d'.format(dim))(in_ch, out_ch, K_D, stride=dim, padding=K_D//2)
        self.bnrm  = getattr(nn, 'BatchNorm{}d'.format(dim))(out_ch)
        self.drop  = nn.Sequential(perm(), nn.Dropout2d(DR_S), perm()) if dim==1 else nn.Dropout2d(DR_S)
        self.block = nn.Sequential(self.conv, nn.ELU(), self.bnrm, self.drop)
    def forward(self, x):
        return self.block(x)

#Residual block class, made up of two convolutional blocks.
class ResBlock(nn.Module):
    def __init__(self, in_ch, hd_ch, out_ch, dim):
        super(ResBlock, self).__init__()
        self.dim  = dim
        self.conv = nn.Sequential(ConvBlock(in_ch, hd_ch, dim), ConvBlock(hd_ch, out_ch, dim))
        self.down = nn.Sequential(nn.MaxPool2d(3, 2, 1), nn.MaxPool2d(3, 2, 1))
        self.skip = getattr(nn, 'Conv{}d'.format(dim))(in_ch, out_ch, 1, bias=False)
    def forward(self, x):
        return self.conv(x) + self.skip(x if self.dim==1 else self.down(x))

#Relational net class, defining architectures for modelling PMPs in Unicode Analogies.
class RelNet(nn.Module):
    def __init__(self, model):
        super(RelNet, self).__init__()
        self.stack    = lambda x : torch.stack([torch.cat((x[:,:8], x[:,i].unsqueeze(1)), dim=1) for i in range(8, 16)], dim=1)
        self.stack_ua = lambda x : torch.stack([torch.cat((x[:,:5], x[:,i].unsqueeze(1)), dim=1) for i in range(5, 9)], dim=1)
        self.model    = model
                
        if model == 'relbase':
            lin_in = S_OC*S_PL
            self.obj_enc = nn.Sequential(ResBlock(   1, F_HC, F_HC, 2), ResBlock(F_HC, F_HC, F_OC, 2))
            self.seq_enc = nn.Sequential(ResBlock(   6, S_OC, S_HC, 1), nn.MaxPool1d(6, 4, 1), 
                                         ResBlock(S_HC, S_HC, S_OC, 1), nn.AdaptiveAvgPool1d(S_PL))            
        elif model in ['resnet', 'blind']:
            lin_in = O_OC*F_PL
            dim_in = 4 if model=='blind' else 6
            self.og_net = nn.Sequential(ResBlock(dim_in, O_HC, O_HC, 2), ResBlock(O_HC, O_HC, O_OC, 2))
        else:
            sys.exit(1)
                    
        self.linear = nn.Sequential(nn.Linear(lin_in, 512), nn.ELU(), nn.BatchNorm1d(512), nn.Dropout(DR_F), 
                      nn.Linear(512, 4 if model=='blind' else 1))

    def forward(self, x):                
        if self.model=='relbase':
            #1. Encode each frame independently.
            x = x.view(-1, 1, 80, 80)
            x = self.obj_enc(x).flatten(1)
            
            #2. Assemble sequences. 
            x = x.view(-1, 9, F_Z)
            x = self.stack_ua(x)
                        
            #3. Extract frame relationships and score sequences.
            x = self.seq_enc(x.view(-1, 6, F_Z)).flatten(1)
            return self.linear(x).view(-1, 4)
        
        elif self.model=='resnet':
            #1. Assemble sequences, extract frame relationships, score sequences.
            x = self.stack_ua(x).view(-1, 6, 80, 80)
            x = self.og_net(x).flatten(1)
            x = self.linear(x).view(-1, 4)
            return x
        
        elif self.model=='blind':
            x = x[:, 5:]
            x = self.og_net(x)
            return self.linear(x.flatten(1))
        
        else:
            sys.exit(1)
            
#Main model class.
class RPM_Solver(BasicModel):
    def __init__(self, model, multi_gpu, lr):
        super(RPM_Solver, self).__init__(model)
        self.model     = model
        self.rel_enc   = nn.DataParallel(RelNet(model), device_ids=[0,1]) if multi_gpu else RelNet(model)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    def compute_loss(self, output, target):
        return F.cross_entropy(output, target)
    
    def forward(self, x):
        return self.rel_enc(x)


# END SCRIPT -------------------------------------------------------------------------------------------------------- #
