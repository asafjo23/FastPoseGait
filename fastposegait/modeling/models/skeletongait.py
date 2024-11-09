import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
from .modules import HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, SetBlockWrapper, \
    conv3x3, conv1x1, BasicBlock2D, BasicBlockP3D
from einops import rearrange

class SkeletonGait(BaseModel):
    def build_network(self, model_cfg):
        in_C, B, C = model_cfg['Backbone']['in_channels'], model_cfg['Backbone']['blocks'], model_cfg['Backbone']['C']
        self.inference_use_emb = model_cfg['use_emb2'] if 'use_emb2' in model_cfg else False

        self.inplanes = 32 * C
        
        # Initial pose processing layer
        self.pose_layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(in_C, self.inplanes, 1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        ))

        # Main backbone layers
        self.layer1 = self.make_layer(BasicBlock2D, 32 * C, stride=[1, 1], blocks_num=B[0], mode='2d')
        self.layer2 = self.make_layer(BasicBlockP3D, 64 * C, stride=[2, 2], blocks_num=B[1], mode='p3d')
        self.layer3 = self.make_layer(BasicBlockP3D, 128 * C, stride=[2, 2], blocks_num=B[2], mode='p3d')
        self.layer4 = self.make_layer(BasicBlockP3D, 256 * C, stride=[1, 1], blocks_num=B[3], mode='p3d')

        # Final layers
        self.FCs = SeparateFCs(16, 256 * C, 128 * C)
        self.BNNecks = SeparateBNNecks(16, 128 * C, class_num=model_cfg['SeparateBNNecks']['class_num'])

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

    def make_layer(self, block, planes, stride, blocks_num, mode='2d'):
        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == '3d':
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], 
                             stride=stride, padding=[0, 0, 0], bias=False),
                    nn.BatchNorm3d(planes * block.expansion))
            elif mode == '2d':
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                    nn.BatchNorm2d(planes * block.expansion))
            elif mode == 'p3d':
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], 
                             stride=[1, *stride], padding=[0, 0, 0], bias=False),
                    nn.BatchNorm3d(planes * block.expansion))
            else:
                raise TypeError('Unknown mode')
        else:
            downsample = lambda x: x

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        s = [1, 1] if mode in ['2d', 'p3d'] else [1, 1, 1]
        for i in range(1, blocks_num):
            layers.append(block(self.inplanes, planes, stride=s))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        pose = ipts[0]  # [N, C, T, V, M]
        pose = pose.transpose(1, 2).contiguous()
        maps = pose[:, :2, ...]
        # Remove M dimension and transpose to match expected input
        # Initial processing
        out0 = self.pose_layer0(maps)  # [N, 32*C, T, V]
        
        # Main backbone
        out1 = self.layer1(out0.squeeze())
        out2 = self.layer2(out1.unsqueeze(-1))
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)  # [n, c, s, h, w]

        # Temporal Pooling
        outs = self.TP(out4, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        
        # Horizontal Pooling
        feat = self.HPP(outs)  # [n, c, p]

        # Final embeddings and logits
        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]

        embed = embed_2 if self.inference_use_emb else embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/pose': {},
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval
