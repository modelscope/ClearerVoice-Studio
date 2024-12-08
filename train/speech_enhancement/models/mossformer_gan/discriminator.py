import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from models.mossformer_gan.conformer import ConformerBlock

class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)

#%% Spectrograms
def segment_specs(y, seg_length=15, seg_hop=4, max_length=None):
    '''
    Segment a spectrogram into "seg_length" wide spectrogram segments.
    Instead of using only the frequency bin of the current time step, 
    the neighboring bins are included as input to the CNN. For example 
    for a seg_length of 7, the previous 3 and the following 3 frequency 
    bins are included.

    A spectrogram with input size [H x W] will be segmented to:
    [W-(seg_length-1) x C x H x seg_length], where W is the width of the 
    original mel-spec (corresponding to the length of the speech signal),
    H is the height of the mel-spec (corresponding to the number of mel bands),
    C is the number of CNN input Channels (always one in our case).
    '''
    if seg_length % 2 == 0:
        raise ValueError('seg_length must be odd! (seg_lenth={})'.format(seg_length))
    if not torch.is_tensor(y):
        y = torch.tensor(y)

    B, _, _ = y.size()
    for b in range(B):
        x = y[b,:,:]
        n_wins = x.shape[1]-(seg_length-1)
        # broadcast magic to segment melspec
        idx1 = torch.arange(seg_length)
        idx2 = torch.arange(n_wins)
        idx3 = idx1.unsqueeze(0) + idx2.unsqueeze(1)
        x = x.transpose(1,0)[idx3,:].unsqueeze(1).transpose(3,2)

        if seg_hop>1:
            x = x[::seg_hop,:]
            n_wins = int(np.ceil(n_wins/seg_hop))

        if max_length is not None:
            if max_length < n_wins:
                raise ValueError('n_wins {} > max_length {}. Increase max window length ms_max_segments!'.format(n_wins, max_length))
            x_padded = torch.zeros((max_length, x.shape[1], x.shape[2], x.shape[3]))
            x_padded[:n_wins,:] = x
            x = x_padded
        if b == 0:
            z = x.unsqueeze(0)
        else:
            z = torch.cat((z, x.unsqueeze(0)), axis = 0)
    B, n, c, f, t = z.size()
    z = z.view(B*n, c, f, t)
    return z

class AdaptCNN(nn.Module):
    '''
    Taken from https://github.com/gabrielmittag/NISQA/blob/master/nisqa/NISQA_lib.py
    --------
    AdaptCNN: CNN with adaptive maxpooling that can be used as framewise model.
    Overall, it has six convolutional layers. This CNN module is more flexible
    than the StandardCNN that requires a fixed input dimension of 48x15.
    '''            
    def __init__(self, 
                 input_channels = 2,
                 c_out_1 = 16, 
                 c_out_2 = 32,
                 c_out_3 = 64,
                 kernel_size = [3,3], 
                 dropout = 0.2,
                 pool_1 = [101, 7],
                 pool_2 = [50, 7],
                 pool_3 = [25, 5],
                 pool_4 = [12, 5],
                 pool_5 = [6, 3],
                 fc_out_h=None,
                 ):
        super().__init__()
        self.name = 'CNN_adapt'

        self.input_channels = input_channels
        self.c_out_1 = c_out_1
        self.c_out_2 = c_out_2
        self.c_out_3 = c_out_3
        self.kernel_size = kernel_size
        self.pool_1 = pool_1
        self.pool_2 = pool_2
        self.pool_3 = pool_3
        self.pool_4 = pool_4
        self.pool_5 = pool_5
        self.dropout_rate = dropout
        self.fc_out_h = fc_out_h

        self.dropout = nn.Dropout2d(p=self.dropout_rate)
        
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
            
        # Set kernel width of last conv layer to last pool width to 
        # downsample width to one.
        self.kernel_size_last = (self.kernel_size[0], self.pool_5[1])
            
        # kernel_size[1]=1 can be used for seg_length=1 -> corresponds to 
        # 1D conv layer, no width padding needed.
        if self.kernel_size[1] == 1:
            self.cnn_pad = (1,0)
        else:
            self.cnn_pad = (1,1)   
            
        self.conv1 = nn.Conv2d(
                self.input_channels,
                self.c_out_1,
                self.kernel_size,
                padding = self.cnn_pad)

        self.bn1 = nn.BatchNorm2d( self.conv1.out_channels )

        self.conv2 = nn.Conv2d(
                self.conv1.out_channels,
                self.c_out_2,
                self.kernel_size,
                padding = self.cnn_pad)

        self.bn2 = nn.BatchNorm2d( self.conv2.out_channels )

        self.conv3 = nn.Conv2d(
                self.conv2.out_channels,
                self.c_out_3,
                self.kernel_size,
                padding = self.cnn_pad)

        self.bn3 = nn.BatchNorm2d( self.conv3.out_channels )

        self.conv4 = nn.Conv2d(
                self.conv3.out_channels,
                self.c_out_3,
                self.kernel_size,
                padding = self.cnn_pad)

        self.bn4 = nn.BatchNorm2d( self.conv4.out_channels )

        self.conv5 = nn.Conv2d(
                self.conv4.out_channels,
                self.c_out_3,
                self.kernel_size,
                padding = self.cnn_pad)

        self.bn5 = nn.BatchNorm2d( self.conv5.out_channels )

        self.conv6 = nn.Conv2d(
                self.conv5.out_channels,
                self.c_out_3,
                self.kernel_size_last,
                padding = (1,0))

        self.bn6 = nn.BatchNorm2d( self.conv6.out_channels )
        
        if self.fc_out_h:
            self.fc = nn.Linear(self.conv6.out_channels * self.pool_3[0], self.fc_out_h)
            self.fan_out = self.fc_out_h
        else:
            self.fan_out = (self.conv6.out_channels * self.pool_3[0])

    def forward(self, x):
        x = F.relu( self.bn1( self.conv1(x) ) )
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_1))

        x = F.relu( self.bn2( self.conv2(x) ) )
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_2))
        
        x = self.dropout(x)
        x = F.relu( self.bn3( self.conv3(x) ) )
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_3))

        x = self.dropout(x)
        x = F.relu( self.bn4( self.conv4(x) ) )
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_4))

        x = self.dropout(x)
        x = F.relu( self.bn5( self.conv5(x) ) )
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_5))

        x = self.dropout(x)
        x = F.relu( self.bn6( self.conv6(x) ) )
        x = x.view(-1, self.conv6.out_channels * self.pool_5[0])
        
        if self.fc_out_h:
            x = self.fc( x ) 
        #print('CNN output: {}'.format(x.shape))
        return x

class PoolAttFF(torch.nn.Module):
    '''
    Taken from https://github.com/gabrielmittag/NISQA/blob/master/nisqa/NISQA_lib.py
    ------
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    '''
    def __init__(self, d_input=384, output_size=1, h=128, dropout=0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_input, h)
        self.linear2 = nn.Linear(h, 1)

        self.linear3 = nn.Linear(d_input, output_size)

        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        att = self.linear2(self.dropout(self.activation(self.linear1(x))))
        att = att.transpose(2,1)
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, x)
        x = x.squeeze(1)
        x = self.linear3(x)

        return x

class Discriminator(nn.Module):
    """Inputs: two waveforms including prediction (x) and groundtruth (y) 
       Outputs: value between 0 ~ 1

       The purpose is to predict the normalized PESQ value btw x and y using a network model
       Process: x and y are concatenated, and inputted to a cnn net, the cnn output is reshaped and
                processed by a self-attention net, the attention output is pooled and sigmoided for 
                final output.
     """
    def __init__(self, ndf, in_channel=2):
        super().__init__()
        self.dim = 384
        self.cnn = AdaptCNN()
        self.att = nn.Sequential(
            ConformerBlock(dim=self.dim, dim_head=self.dim//4, heads=4,
                                             conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2),
            ConformerBlock(dim=self.dim, dim_head=self.dim//4, heads=4,
                                             conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)
        )
        self.pool = PoolAttFF()
        self.sigmoid = LearnableSigmoid(1)

    def forward(self, x, y):
        B,_,_,_ = x.size()
        x = segment_specs(x.squeeze(1))
        y = segment_specs(y.squeeze(1))
        xy = torch.cat([x, y], dim=1)
        cnn_out = self.cnn(xy)
        _, d = cnn_out.size()
        cnn_out = cnn_out.view(B,-1,d)
        att_out = self.att(cnn_out)
        pool_out = self.pool(att_out)        
        out = self.sigmoid(pool_out)
        return out
    
