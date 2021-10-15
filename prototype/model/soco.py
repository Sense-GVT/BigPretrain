import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os
from .resnet import resnet50

import spring.linklink as link
from spring.linklink.nn import SyncBatchNorm2d
from spring.linklink.nn import syncbnVarMode_t
import torchvision
from prototype.utils.dist import simple_group_split

BN = None
__all__ = ['clip_res50_image_simsiam_2_view']

class AllGather(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, tensor):
        ctx.rank = link.get_rank()
        ctx.world_size = link.get_world_size()

        y = tensor.new(ctx.world_size, *tensor.size())
        link.allgather(y, tensor)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        in_grad = torch.zeros_like(grad_output)
        in_grad.copy_(grad_output)
        # sum grad for gathered tensor
        link.allreduce(in_grad)
        # split
        return in_grad[ctx.rank]

def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

class projection_MLP(nn.Module):
    def __init__(self, in_dim, hid_dim=4096, out_dim=256, num_layers=2):
        super(projection_MLP, self).__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out-
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d.
        This MLP has 3 layers.
        '''
        self.num_layers = num_layers

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.linear1 = nn.Linear(in_dim, hid_dim)
        # self.bn1 = nn.BatchNorm1d(hid_dim)
        self.bn1 = BN(hid_dim)
        self.relu1 = nn.ReLU(inplace=True)

        if self.num_layers > 1:
            self.linear2 = nn.Linear(hid_dim, out_dim)
            
        if self.num_layers > 2:
            # self.bn2 = nn.BatchNorm1d(out_dim)
            self.bn2 = BN(out_dim)
            self.relu2 = nn.ReLU(inplace=True)
            self.linear3 = nn.Linear(out_dim, out_dim)
            # self.bn3 = nn.BatchNorm1d(out_dim)
            self.bn3 = BN(out_dim)

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        # b, _ = x.shape
        # import pdb
        # pdb.set_trace()
        # layer 1
        x = self.linear1(x)
        
        # layer 2
        if self.num_layers == 2:
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.linear2(x)
            
        if self.num_layers == 3:
            x = self.bn2(x)
            x = self.relu2(x)
            # layer 3
            x = self.linear3(x)
            # x.reshape(b, self.out_dim, 1)
            x = self.bn3(x)
            # x.reshape(b, self.out_dim)

        return x

class prediction_MLP(nn.Module):
    def __init__(self, in_dim, hid_dim=4096, out_dim=256,num_layers=2): # bottleneck structure
        super(prediction_MLP, self).__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers.
        The dimension of h’s input and output (z and p) is d = 2048,
        and h’s hidden layer’s dimension is 512, making h a
        bottleneck structure (ablation in supplement).
        '''
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.linear1 = nn.Linear(in_dim, hid_dim)
        if self.num_layers == 2:
            # self.bn1 = nn.BatchNorm1d(hid_dim)
            self.bn1 = BN(hid_dim)
            self.relu1 = nn.ReLU(inplace=True)

            self.layer2 = nn.Linear(hid_dim, out_dim)
            """
            Adding BN to the output of the prediction MLP h does not work
            well (Table 3d). We find that this is not about collapsing.
            The training is unstable and the loss oscillates.
            """

    def forward(self, x):
        b, _ = x.shape

        # layer 1
        x = self.linear1(x)
        if self.num_layers == 2:
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.layer2(x)
        return x
  
def R50_SOCO(**kwargs):
    image_encode = resnet50(**kwargs['image_encode'])
    model = SOCO_pipline(image_encode, [256, 512, 1024, 2048], **kwargs['image_encode'])
    return model


class SOCO_pipline(nn.Module):
    """
    image_encode (nn.Module): backbone
    fpn_sizes (list): channel dimension of P2~P5
    """
    def __init__(self, image_encode, fpn_sizes, **kwargs):
        super().__init__()
        global BN
        use_sync_bn = kwargs['use_sync_bn']
        bn_group_size = kwargs['bn_group_size']
        bn_var_mode = syncbnVarMode_t.L2
        bn_sync_stats = kwargs['bn_sync_stats']
        rank = link.get_rank()
        world_size = link.get_world_size()
        bn_group = simple_group_split(world_size, rank, world_size // bn_group_size)

        def BNFunc(*args, **kwargs):
            return SyncBatchNorm2d(*args, **kwargs,
                                   group=bn_group,
                                   sync_stats=bn_sync_stats,
                                   var_mode=bn_var_mode)

        if use_sync_bn:
            BN = BNFunc
            print('[Rank {}] use SyncBatchNorm in bn_group {}'.format(rank, bn_group), flush=True)
        else:
            BN = nn.BatchNorm2d

        self.backbone = image_encode
        self.use_predictor = True
        self.freeze_backbone = True 
        self.freeze_at = 0
        self.moco = kwargs['momentum']
        print('load pretrain from:', kwargs['pretrained'])
        if kwargs['pretrained'] is not None and kwargs['pretrained'] != 'None':
            model_dict = self.backbone.state_dict()
            pretrain_dict=torch.load(kwargs['pretrained'], 'cpu')
            tmp_dict = {}
            if 'state_dict' in pretrain_dict.keys():
                pretrain_dict = pretrain_dict['state_dict']
                for key, val in pretrain_dict.items():
                    if 'num_batches_tracked' not in key:
                        tmp_dict[key] = val
            elif 'model' in pretrain_dict.keys():
                pretrain_dict = pretrain_dict['model']
                for key, val in pretrain_dict.items():
                    if 'backbone' in key and 'num_batches_tracked' not in key:
                        real_key = '.'.join(key.split('.')[1:])
                        tmp_dict[real_key] = val
            else:
                assert False
            missing_keys, unexpected_keys = self.backbone.load_state_dict(tmp_dict, strict=False)
            print('missing_keys:', missing_keys, '\n', 'unexpected_keys:', unexpected_keys, flush=True)
        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3])
        self.roi_output_size = kwargs['roi_size']
        self.head = TwoMLPHead(256*self.roi_output_size*self.roi_output_size,1024)
        self.projector_s = projection_MLP(1024,hid_dim=4096, out_dim=256)
        if self.use_predictor:
            self.predictor_s = prediction_MLP(256, hid_dim=4096, out_dim=256)

    def forward(self, input, proposal=None, fpn_level=None, return_fea=False,use_rpn=False):
        output = {}
        if self.freeze_backbone:
            self.backbone.eval()
            if self.freeze_at == 4:
                self.backbone.layer4.train()
        else:
            self.backbone.train() 

        fea = self.backbone(input, self.freeze_backbone, self.freeze_at)
    
        fpn_features = self.fpn([fea[0], fea[1], fea[2], fea[3]])

        res_list = []
        for level in range(4):
            fpn_feature = fpn_features[level]
            scale = 4 * pow(2, level)
            if isinstance(proposal, list):
                proposal_bbox = [b/scale for b in proposal]
            else:
                proposal_bbox = proposal[:, 1:5]/scale
            roi_feature = torchvision.ops.roi_align(fpn_feature, proposal_bbox, [self.roi_output_size,self.roi_output_size])   
            res_list.append(roi_feature)

        res_list = torch.cat(res_list, 0)
        roi_feature_index = []

        for i,pbatch in enumerate(proposal): 
            for j in range(len(pbatch)):
                roi_feature_index.append(fpn_level[i][0][j]*len(res_list)//4 + len(pbatch)*i + j)

        roi_feature_index = torch.tensor(roi_feature_index).to(res_list.device)
        roi_features = torch.index_select(res_list, 0, roi_feature_index)
        
        head_fea = self.head(roi_features)
        if return_fea:
            output['head_fea'] = head_fea
        z1s = self.projector_s(head_fea)
        if self.use_predictor:
            p1s = self.predictor_s(z1s)
            output['p1s'] = p1s
        else:
            output['z1s'] = z1s

        return output


class PyramidFeatures(nn.Module):
    """
    P2~P5 FPN
    Arguments:
        C2_size, C3_size, C4_size, C5_size (int): channel dimension
        feature_size (int): output channel dimension
    """

    def __init__(self, C2_size, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P3 elementwise to C2
        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        C2, C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1(C2)
        P2_x = P2_x + P3_upsampled_x
        P2_x = self.P2_2(P2_x)

        return [P2_x, P3_x, P4_x, P5_x]


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models
    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, out_channels=1024):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, out_channels)
        self.fc7 = nn.Linear(out_channels, out_channels)

    def forward(self, x):

        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class Momentum(nn.Module):
    """
    encoder_q (nn.Module): online network
    encoder_k (nn.Module): target network
    freeze_set: frozen backbone settting
    K: queue size; number of negative keys (default: 65536)
    m: moco momentum of updating key encoder (default: 0.999)
    T: softmax temperature (default: 0.07)
    total_step: use for linear-decay moco momentum
    """

    def __init__(self, encoder_q, encoder_k, freeze_set=None, K=65536, m=0.99, T=0.07, 
                 group_size=None, total_step=125001):
        
        super(Momentum, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.total_step = total_step
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        # update encoder_q and encoder_k
        self.encoder_q.use_predictor = True
        self.encoder_k.use_predictor = False
        self.encoder_q.freeze_backbone = freeze_set.freeze_backbone
        self.encoder_k.freeze_backbone = freeze_set.freeze_backbone
        self.encoder_q.freeze_at = freeze_set.freeze_at
        self.encoder_k.freeze_at = freeze_set.freeze_at
        self.freeze_backbone = freeze_set.freeze_backbone
        self.freeze_at = freeze_set.freeze_at

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        rank = link.get_rank()
        world_size = link.get_world_size()
        self.group_size = world_size if group_size is None else min(
            world_size, group_size)

        assert world_size % self.group_size == 0
        self.group_idx = simple_group_split(
            world_size, rank, world_size // self.group_size)

    @torch.no_grad()
    def _momentum_update_key_encoder(self,step=None):
        if step is not None:
            self.m = min(0.99 + step/self.total_step*(1-0.99),1)
        else:
            self.m = 0.999
        for param_q, param_k, (param_q_name,_), (param_k_name,_) in zip(self.encoder_q.parameters(), self.encoder_k.parameters(), self.encoder_q.named_parameters(), self.encoder_k.named_parameters()):
            if self.freeze_backbone:
                if self.freeze_at == 4:
                    if 'layer4' in param_q_name or 'fpn' in param_q_name or 'head' in param_q_name or 'projector' in param_q_name:
                        param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
                else:
                    if 'fpn' in param_q_name or 'head' in param_q_name or 'projector' in param_q_name:
                        param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            else:
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        

    def forward(self, input, step=None):
        output = {}
        im_q, im_k1, im_k2 = input['view1'], input['view2'], input['view3']
        im_q = im_q.contiguous()
        im_k1 = im_k1.contiguous()
        im_k2 = im_k2.contiguous()

        online_q1 = self.encoder_q(im_q, input['proposal_bbox1'],input['fpn_level1'],
                                    return_fea=True)

        online_k1 = self.encoder_q(im_k1, input['proposal_bbox2'],input['fpn_level2'])  # queries: NxC
        online_k2 = self.encoder_q(im_k2, input['proposal_bbox3'],input['fpn_level3'])  # queries: NxC
        # update the key encoder
        self._momentum_update_key_encoder(step=step)
        # compute key features
        with torch.no_grad():  # no gradient to keys
            target_q1 = self.encoder_k(im_q, input['proposal_bbox1'],input['fpn_level1'])
            target_k1 = self.encoder_k(im_k1, input['proposal_bbox2'],input['fpn_level2'])  # keys: NxC
            target_k2 = self.encoder_k(im_k2, input['proposal_bbox3'],input['fpn_level3'])  # keys: NxC

        output['online_q1'] = online_q1['p1s']
        output['target_k1'] = target_k1['z1s']
        output['target_k2'] = target_k2['z1s']
        output['target_q1'] = target_q1['z1s']
        output['online_k1'] = online_k1['p1s']
        output['online_k2'] = online_k2['p1s']
        return output

