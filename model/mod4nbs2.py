from os import name
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
#from PIL import Image, ImageOps

BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d


class SpatialGCN(nn.Module):
    def __init__(self, plane):
        super(SpatialGCN, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(inter_plane)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1))#,BatchNorm2d(plane) BN1

    def forward(self, x):
        # b, c, h, w = x.size()
        node_k = self.node_k(x)
        node_v = self.node_v(x)
        node_q = self.node_q(x)
        b,c,h,w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)
        node_q = node_q.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)
        # A = k * q
        # AV = k * q * v
        # AVW = k *(q *v) * w
        AV = torch.bmm(node_q,node_v)
        AV = self.softmax(AV)
        AV = torch.bmm(node_k, AV)
        AV = AV.transpose(1, 2).contiguous()
        AVW = self.conv_wg(AV)
        AVW = self.bn_wg(AVW)
        AVW = AVW.view(b, c, h, -1)
        out = F.relu_(self.out(AVW) + x)
        return out


class ChannelGCN(nn.Module):
    def __init__(self, planes, ratio=4):
        super(ChannelGCN, self).__init__()
        self.phi = nn.Conv2d(planes, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_phi = BatchNorm2d(planes // ratio * 2)
        self.theta = nn.Conv2d(planes, planes // ratio, kernel_size=1, bias=False)
        self.bn_theta = BatchNorm2d(planes // ratio)

        #  Interaction Space
        #  Adjacency Matrix: (-)A_g
        self.conv_adj = nn.Conv1d(planes // ratio, planes // ratio, kernel_size=1, bias=False)
        self.bn_adj = BatchNorm1d(planes // ratio)

        #  State Update Function: W_g
        self.conv_wg = nn.Conv1d(planes // ratio * 2, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(planes // ratio * 2)

        #  last fc
        self.conv3 = nn.Conv2d(planes // ratio * 2, planes, kernel_size=1, bias=False)
        #self.bn3 = BatchNorm2d(planes)  #BN2
    
    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return x

    def forward(self, x):
        x_sqz, b = x, x

        x_sqz = self.phi(x_sqz)
        x_sqz = self.bn_phi(x_sqz)
        x_sqz = self.to_matrix(x_sqz)

        b = self.theta(b)
        b = self.bn_theta(b)
        b = self.to_matrix(b)

        # Project
        z_idt = torch.matmul(x_sqz, b.transpose(1, 2))

        # # # # Interaction Space # # # #
        z = z_idt.transpose(1, 2).contiguous()

        z = self.conv_adj(z)
        z = self.bn_adj(z)

        z = z.transpose(1, 2).contiguous()
        # Laplacian smoothing: (I - A_g)Z => Z - A_gZ
        z += z_idt

        z = self.conv_wg(z)
        z = self.bn_wg(z)

        # # # # Re-projection Space # # # #
        # Re-project
        y = torch.matmul(z, b)

        n, _, h, w = x.size()
        y = y.view(n, -1, h, w)

        y = self.conv3(y)
        #y = self.bn3(y)

        g_out = F.relu_(x+y)

        return g_out


class DualGCN(nn.Module):
    """
        Feature GCN with coordinate GCN
    """
    def __init__(self, planes, ratio=4):
        super(DualGCN, self).__init__()

        self.phi = nn.Conv2d(planes, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_phi = BatchNorm2d(planes // ratio * 2)
        self.theta = nn.Conv2d(planes, planes // ratio, kernel_size=1, bias=False)
        self.bn_theta = BatchNorm2d(planes // ratio)

        #  Interaction Space
        #  Adjacency Matrix: (-)A_g
        self.conv_adj = nn.Conv1d(planes // ratio, planes // ratio, kernel_size=1, bias=False)
        self.bn_adj = BatchNorm1d(planes // ratio)

        #  State Update Function: W_g
        self.conv_wg = nn.Conv1d(planes // ratio * 2, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(planes // ratio * 2)

        #  last fc
        self.conv3 = nn.Conv2d(planes // ratio * 2, planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes)

        self.local = nn.Sequential(
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes))
        self.gcn_local_attention = SpatialGCN(planes)

        self.final = nn.Sequential(nn.Conv2d(planes * 2, planes, kernel_size=1, bias=False),
                                   BatchNorm2d(planes))

    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return x

    def forward(self, feat):
        # # # # Local # # # #
        x = feat
        local = self.local(feat)
        local = self.gcn_local_attention(local)
        local = F.interpolate(local, size=x.size()[2:], mode='bilinear', align_corners=True)
        spatial_local_feat = x * local + x

        # # # # Projection Space # # # #
        x_sqz, b = x, x

        x_sqz = self.phi(x_sqz)
        x_sqz = self.bn_phi(x_sqz)
        x_sqz = self.to_matrix(x_sqz)

        b = self.theta(b)
        b = self.bn_theta(b)
        b = self.to_matrix(b)

        # Project
        z_idt = torch.matmul(x_sqz, b.transpose(1, 2))

        # # # # Interaction Space # # # #
        z = z_idt.transpose(1, 2).contiguous()

        z = self.conv_adj(z)
        z = self.bn_adj(z)

        z = z.transpose(1, 2).contiguous()
        # Laplacian smoothing: (I - A_g)Z => Z - A_gZ
        z += z_idt

        z = self.conv_wg(z)
        z = self.bn_wg(z)

        # # # # Re-projection Space # # # #
        # Re-project
        y = torch.matmul(z, b)

        n, _, h, w = x.size()
        y = y.view(n, -1, h, w)

        y = self.conv3(y)
        y = self.bn3(y)

        g_out = F.relu_(x+y)

        # cat or sum, nearly the same results
        out = self.final(torch.cat((spatial_local_feat, g_out), 1))

        return out


class DualGCN_parallel(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, stride=1):
        super(DualGCN_parallel, self).__init__()
        self.conva = nn.Sequential(nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
                                   BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))
        self.dualgcn = DualGCN(interplanes)
        self.convb = nn.Sequential(nn.Conv2d(interplanes, interplanes, 3, padding=1, bias=False),
                                   BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inplanes + interplanes, interplanes, kernel_size=3, stride=stride, padding=1, dilation=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
            nn.Conv2d(interplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        output = self.conva(x)
        output = self.dualgcn(output)
        output = self.convb(output)
        output = self.bottleneck(torch.cat([x, output], 1))
        return output


class DualGCN_Spatial_fist(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, stride=1):
        super(DualGCN_Spatial_fist, self).__init__()
        self.local = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, 3, groups=inplanes, stride=2, padding=1, bias=False),
            BatchNorm2d(inplanes),
            nn.Conv2d(inplanes, inplanes, 3, groups=inplanes, stride=2, padding=1, bias=False),
            BatchNorm2d(inplanes),
            nn.Conv2d(inplanes, inplanes, 3, groups=inplanes, stride=2, padding=1, bias=False),
            BatchNorm2d(inplanes))

        self.gcn_local_attention = SpatialGCN(inplanes)
        
        self.conva = nn.Sequential(nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
                                   BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))

        
        self.gcn_feature_attention = ChannelGCN(interplanes)

        self.convb = nn.Sequential(nn.Conv2d(interplanes, interplanes, 3, padding=1, bias=False),
                                   BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(interplanes, interplanes, kernel_size=3, stride=stride, padding=1, dilation=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
            nn.Conv2d(interplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        # spatial part
        local = self.local(x)
        local = self.gcn_local_attention(local)
        local = F.interpolate(local, size=x.size()[2:], mode='bilinear', align_corners=True)
        spatial_local_feat = x * local + x

        # channel part
        CG_part = self.conva(spatial_local_feat)
        CG_part = self.gcn_feature_attention(CG_part)
        CG_part = self.convb(CG_part)

        # output
        output = self.bottleneck(CG_part)

        return output


class DualGCN_Channel_fist(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, stride=1):
        super(DualGCN_Channel_fist, self).__init__()
        self.conva = nn.Sequential(nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
                                   BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))
        self.gcn_feature_attention = ChannelGCN(interplanes)
        self.convb = nn.Sequential(nn.Conv2d(interplanes, interplanes, 3, padding=1, bias=False),
                                   BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))

        self.local = nn.Sequential(
            nn.Conv2d(interplanes, interplanes, 3, groups=interplanes, stride=2, padding=1, bias=False),
            BatchNorm2d(interplanes),
            nn.Conv2d(interplanes, interplanes, 3, groups=interplanes, stride=2, padding=1, bias=False),
            BatchNorm2d(interplanes),
            nn.Conv2d(interplanes, interplanes, 3, groups=interplanes, stride=2, padding=1, bias=False),
            BatchNorm2d(interplanes))
        self.gcn_local_attention = SpatialGCN(interplanes)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(interplanes, interplanes, kernel_size=3, stride=stride, padding=1, dilation=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
            nn.Conv2d(interplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        # channel part
        CG_part = self.conva(x)
        CG_part = self.gcn_feature_attention(CG_part)
        CG_part = self.convb(CG_part)

        # spatial part
        local = self.local(CG_part)
        local = self.gcn_local_attention(local)
        local = F.interpolate(local, size=x.size()[2:], mode='bilinear', align_corners=True)
        spatial_local_feat = CG_part * local + CG_part

        # output
        output = self.bottleneck(spatial_local_feat)

        return output


class fu_DualGCN_Spatial_fist(nn.Module):
    def __init__(self, inchannels):
        super(fu_DualGCN_Spatial_fist, self).__init__()
        self.sGCN = SpatialGCN(inchannels)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, padding=1, dilation=1),
            nn.ReLU(inchannels)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, padding=1, dilation=1),
            nn.ReLU(inchannels)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, padding=3, dilation=3),
            nn.ReLU(inchannels)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, padding=3, dilation=3),
            nn.ReLU(inchannels)
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(inchannels*5, inchannels, 1, padding=0),
            nn.ReLU(inchannels)
        )
        self.cGCN = ChannelGCN(inchannels)
    
    def forward(self, x):
        F_sGCN = self.sGCN(x)
        conv1 = self.conv_1(F_sGCN)
        conv2 = self.conv_2(conv1)
        conv3 = self.conv_3(F_sGCN)
        conv4 = self.conv_4(conv3)

        F_DCM = self.conv_5(torch.cat([F_sGCN, conv1, conv2, conv3, conv4], dim=1))
        F_cGCN = self.cGCN(F_DCM)
        F_unit = F_cGCN + x
        return F_unit

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None, pad_model=None,groups=1):
        super(ConvBlock, self).__init__()

        self.pad_model = pad_model
        self.norm = norm
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(self.output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(self.output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        
        if self.pad_model == None:   
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, self.padding, groups=groups,bias=self.bias)
        elif self.pad_model == 'reflection':
            self.padding = nn.Sequential(nn.ReflectionPad2d(self.padding))
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, 0,groups=groups,bias=self.bias)

    def forward(self, x):
        out = x
        if self.pad_model is not None:
            out = self.padding(out)

        if self.norm is not None:
            out = self.bn(self.conv(out))
        else:
            out = self.conv(out)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class ResnetBlock(nn.Module):
    def __init__(self, input_size, kernel_size=3, stride=1, padding=1, bias=True, scale=1, activation='prelu', norm='batch', pad_model=None,groups=1):
        super().__init__()

        self.norm = norm
        self.pad_model = pad_model
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.scale = scale
        
        if self.norm =='batch':
            self.normlayer = torch.nn.BatchNorm2d(input_size)
        elif self.norm == 'instance':
            self.normlayer = torch.nn.InstanceNorm2d(input_size)
        else:
            self.normlayer = None

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        else:
            self.act = None

        if self.pad_model == None:   
            self.conv1 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding, bias=bias,groups=groups)
            self.conv2 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, padding, bias=bias)
            self.pad = None
        elif self.pad_model == 'reflection':
            self.pad = nn.Sequential(nn.ReflectionPad2d(padding))
            self.conv1 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, 0, bias=bias,groups=groups)
            self.conv2 = torch.nn.Conv2d(input_size, input_size, kernel_size, stride, 0, bias=bias)

        layers = filter(lambda x: x is not None, [self.pad, self.conv1, self.normlayer, self.act, self.pad,self.conv2, self.normlayer, self.act])#
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = x
        out = self.layers(x)
        out = out * self.scale
        out = torch.add(out, residual)
        return out

class CGConv(nn.Module):
    ''' Mutual Affine Convolution (MAConv) layer '''
    def __init__(self, in_channels, out_channels,bias=True,g_inchannels=3):  #
        super(CGConv, self).__init__()
        self.split=in_channels
        inter_channels=in_channels
        setattr(self, 'fc{}', nn.Sequential(*[
            nn.Conv2d(in_channels=g_inchannels, out_channels=inter_channels, 
                        kernel_size=1, stride=1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            #nn.PReLU(init=0.5),
            nn.Conv2d(in_channels=inter_channels, out_channels=inter_channels*2, 
                        kernel_size=1, stride=1, padding=0, bias=bias),
        ]))
        setattr(self, 'conv{}', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                                    kernel_size=3, stride=1, padding=1, bias=bias))
    def forward(self,input,g_input):
        scale, translation = torch.split(getattr(self, 'fc{}')(g_input), (self.split,self.split), dim=1)
        output=getattr(self, 'conv{}')(F.relu(input*torch.sigmoid(scale) + translation))
        return output

class Bandmodulation(nn.Module):
    def __init__(self,in_channels, out_channels,f_channels=4,bias=True,split=4,copy_wight=3,g_inchannels=1):
        super(Bandmodulation,self).__init__()
        self.shortcut=True
        if in_channels !=out_channels:
             self.shortcut=False
        self.num_split=split
        self.copy_wight=copy_wight
        self.split_list=[] #特征图拆分
        self.fusion_list=[] #融合图拆分
        in_split,f_split=in_channels//split,f_channels//split
        out_split=out_channels//split
        for i in range(self.num_split):
            setattr(self, 'conv{}'.format(i), nn.Conv2d(in_channels=in_split+f_split, out_channels=in_split, 
                                                        kernel_size=1, stride=1, padding=0, bias=bias))
            setattr(self, 'cgconv{}'.format(i),CGConv(in_channels=in_split,out_channels=out_split,g_inchannels=copy_wight*g_inchannels))
            self.split_list.append(in_split)
            self.fusion_list.append(f_split)
    def forward(self,input,f_input,g_input):
        t_input = torch.split(input, self.split_list, dim=1) #tuple (4,[1,1,128,128])
        t_fusion= torch.split(f_input, self.fusion_list, dim=1)
        output = [] 
        g_x=g_input
        for i in range(self.copy_wight-1):
            g_x=torch.cat((g_x,g_input),1)

        for i in range(self.num_split):
            split_fusion=getattr(self,'conv{}'.format(i))(torch.cat((t_input[i],t_fusion[i]),1)) #特征融合
            output.append(getattr(self,'cgconv{}'.format(i))(split_fusion,g_x))
        if not self.shortcut:
            return torch.cat(output,1)
        return torch.cat(output, 1)+input


class ab_DualGCN_Spatial_fist(nn.Module):
    def __init__(self, inchannels,f_channels=4,copy_wight=1,g_inchannels=1):
        super(ab_DualGCN_Spatial_fist, self).__init__()
        self.BMM=Bandmodulation(inchannels,inchannels,f_channels=f_channels,copy_wight=copy_wight,g_inchannels=g_inchannels)#pan0   pan 3,1
        self.sGCN = SpatialGCN(inchannels)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, padding=1, dilation=1),
            nn.ReLU(inchannels)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, padding=1, dilation=1),
            nn.ReLU(inchannels)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, padding=3, dilation=3),
            nn.ReLU(inchannels)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, padding=3, dilation=3),
            nn.ReLU(inchannels)
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(inchannels*5, inchannels, 1, padding=0),
            nn.ReLU(inchannels)
        )
        #self.cGCN = ChannelGCN(inchannels)
        n_resblocks = 0
        n_resblocks1 = 1
        res_block_s1 = [
            ConvBlock(inchannels, 24, 3, 1, 1, activation='prelu', norm=None, bias = False,groups=1), #None
        ]
        for i in range(n_resblocks1):
            res_block_s1.append(ResnetBlock(24, 3, 1, 1, 0.1, activation='prelu', norm=None,groups=1))
        self.res_block_s1 = nn.Sequential(*res_block_s1)
        res_block_s2 = [
            ConvBlock(inchannels, 24, 1, 1, 0, activation='prelu', norm=None, bias = False),
        ]
        for i in range(n_resblocks):
            res_block_s2.append(ResnetBlock(24, 3, 1, 1, 0.1, activation='prelu', norm=None))#None
        self.res_block_s2 = nn.Sequential(*res_block_s2)
        res_block_s3 = [
            ConvBlock(inchannels, 24, 3, 1, 1, activation='prelu', norm=None, bias = False),
        ]
        for i in range(n_resblocks):
            res_block_s3.append(ResnetBlock(24, 3, 1, 1, 0.1, activation='prelu', norm=None))
        self.res_block_s3 = nn.Sequential(*res_block_s3)

    def forward(self,x,f_x,g_x):
        #F_sGCN = self.res_block_s1(x)+x
        F_sGCN =self.BMM(x,f_x,g_x)
        F_sGCN = self.res_block_s1(F_sGCN)
        #F_sGCN = self.sGCN(x)
        #F_sGCN=x
        conv1 = self.conv_1(F_sGCN)
        conv2 = self.conv_2(conv1)
        conv3 = self.conv_3(F_sGCN)
        conv4 = self.conv_4(conv3)

        F_DCM = self.conv_5(torch.cat([F_sGCN, conv1, conv2, conv3, conv4], dim=1))
        #F_DCM=self.res_block_s3(F_sGCN)
        #F_cGCN=F_DCM
        #F_cGCN = self.cGCN(F_DCM)
        F_cGCN = self.res_block_s2(F_DCM)
        #F_cGCN=self.BMM(F_DCM,g_x)
        F_unit = F_cGCN + x
        return F_unit

class ab_module(nn.Module):
    def __init__(self, inchannels,copy_wight=1,g_inchannels=4):
        super(ab_module, self).__init__()
        self.BMM=Bandmodulation(inchannels,inchannels,copy_wight=copy_wight,g_inchannels=g_inchannels)#pan0   pan 3,1
        self.sGCN = SpatialGCN(inchannels)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, padding=1, dilation=1),
            nn.ReLU(inchannels)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, padding=1, dilation=1),
            nn.ReLU(inchannels)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, padding=3, dilation=3),
            nn.ReLU(inchannels)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, padding=3, dilation=3),
            nn.ReLU(inchannels)
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(inchannels*5, inchannels, 1, padding=0),
            nn.ReLU(inchannels)
        )
        #self.cGCN = ChannelGCN(inchannels)
    
        n_resblocks = 1
        res_block_s1 = [
            ConvBlock(inchannels, 24, 3, 1, 1, activation='prelu', norm=None, bias = False), #None
        ]
        for i in range(n_resblocks):
            res_block_s1.append(ResnetBlock(24, 3, 1, 1, 0.1, activation='prelu', norm=None))
        self.res_block_s1 = nn.Sequential(*res_block_s1)
        res_block_s2 = [
            ConvBlock(inchannels, 24, 1, 1, 0, activation='prelu', norm=None, bias = False),
        ]
        #for i in range(n_resblocks):
        #    res_block_s2.append(ResnetBlock(24, 3, 1, 1, 0.1, activation='prelu', norm=None))#None
        self.res_block_s2 = nn.Sequential(*res_block_s2)

    def forward(self, x,g_x):
        F_sGCN = self.res_block_s1(x)
        #F_sGCN =self.BMM(x,g_x)
        #F_sGCN = self.sGCN(x)
        #F_sGCN=x
        conv1 = self.conv_1(F_sGCN)
        conv2 = self.conv_2(conv1)
        conv3 = self.conv_3(F_sGCN)
        conv4 = self.conv_4(conv3)

        F_DCM = self.conv_5(torch.cat([F_sGCN, conv1, conv2, conv3, conv4], dim=1))
        #F_DCM=self.res_block_s3(F_sGCN)
        #F_cGCN=F_DCM
        #F_cGCN = self.cGCN(F_DCM)
        F_cGCN = self.res_block_s2(F_DCM)
        #F_cGCN=self.BMM(F_DCM,g_x)
        F_unit = F_cGCN + x
        return F_unit

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, args): #, base_filter, args
        super(Net, self).__init__()
        inchannel=num_channels*2
        interplanes=inchannel*3 #24
        self.head = ConvBlock(inchannel, interplanes, 9, 1, 4, activation='prelu', norm=None, bias = False)
        self.head2=ConvBlock(1, 3, 9, 1, 4, activation='prelu', norm=None, bias = False)
        self.conv1=ConvBlock(24,24 , 5, 1, 2, activation='prelu', norm=None, bias = False)

        self.gcn_basic1=ab_DualGCN_Spatial_fist(24)
        self.gcn_basic2=ab_DualGCN_Spatial_fist(24)
        #self.gcn_basic0=ab_DualGCN_Spatial_fist(24,copy_wight=3,g_inchannels=1)
        self.gconv1=ConvBlock(24, 24, 3, 1, 1, activation='prelu', norm=None, bias = False,groups=4)
        self.gconv2=ConvBlock(24, 24, 3, 1, 1, activation='prelu', norm=None, bias = False,groups=4)

        self.conv2=ConvBlock(24, 24, 3, 1, 1, activation='prelu', norm=None, bias = False)
        self.output_conv = ConvBlock(24, num_channels, 5, 1, 2, activation=None, norm=None, bias = False)
    def forward(self,l_ms,bms,pan):
        
        pan0=self.head2(pan)
        pan0=torch.cat((pan, pan0), 1) #128  c=4
        cbms=torch.cat((bms, pan0), 1) #128 c=8

        x0=self.head(cbms) #128 c=16
        x1=self.conv1(x0)  #24

        x=self.gcn_basic1(x1,bms,pan)#24   pan0
        x=self.gcn_basic2(x,bms,pan) #24


        x=torch.add(x,x1) #c=24
        x=self.gconv1(x)
        x=self.conv2(x)  #c=24
        x=torch.add(x,x0) #128   c=24
        x=self.gconv2(x)
        x=self.output_conv(x)+bms
        #x=F.interpolate(x, scale_factor=2, mode='bicubic')        
        return x


if __name__ == "__main__":

    from torchvision.transforms import Compose, ToTensor
    def transform():
        return Compose([
            ToTensor(),
        ])


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model=Net(4)  #, map_location=torch.device('cpu')
    model.eval()
    #print(model)
    #   ms:torch.Size([4, 128, 128])   pan:([1, 128, 128])
    img=torch.ones((1,4,128,128))
    l_ms=torch.ones((1,4,32,32))
    pan=torch.ones((1,1,128,128))
    output_end=model(l_ms,img,pan)
    print(output_end.shape)
    #print(output_mid.shape)