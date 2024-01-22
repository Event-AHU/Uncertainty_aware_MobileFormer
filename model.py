import time
import torch
import torch.nn as nn

from torch.nn import init
from utils.mobile import Mobile, hswish, MobileDown
from utils.former import Former
from utils.bridge import Mobile2Former, Former2Mobile
from utils.config import config_294, config_508, config_52

import pdb
class BaseBlock(nn.Module):
    def __init__(self, inp, exp, out, se, stride, heads, dim):
        super(BaseBlock, self).__init__()
        if stride == 2:
            self.mobile = MobileDown(3, inp, exp, out, se, stride, dim)
        else:
            self.mobile = Mobile(3, inp, exp, out, se, stride, dim)
        self.mobile2former = Mobile2Former(dim=dim, heads=heads, channel=inp)
        self.former = Former(dim=dim)
        self.former2mobile = Former2Mobile(dim=dim, heads=heads, channel=out)

    def forward(self, inputs):
        x, z = inputs # b c t h w [8, 16, 4, 112, 112]/ b n c [8, 6, 192]
        # x [12, 96, 1, 14, 14]
        # z [12,6,192]
        z_hid = self.mobile2former(x, z) # [8, 6, 192]
        z_out = self.former(z_hid)
        x_hid = self.mobile(x, z_out) # [8, 24, 2, 56, 56]
        x_out = self.former2mobile(x_hid, z_out)
        return [x_out, z_out]
# config_294 = {
#     'name': 'mf294',
#     'token': 6,  # tokens
#     'embed': 192,  # embed_dim
#     'stem': 16,
#     # stage1
#     'bneck': {'e': 32, 'o': 16, 's': 1},  # exp out stride
#     'body': [
#         # stage2
#         {'inp': 16, 'exp': 96, 'out': 24, 'se': None, 'stride': 2, 'heads': 2},
#         {'inp': 24, 'exp': 96, 'out': 24, 'se': None, 'stride': 1, 'heads': 2},
#         # stage3
#         {'inp': 24, 'exp': 144, 'out': 48, 'se': None, 'stride': 2, 'heads': 2},
#         {'inp': 48, 'exp': 192, 'out': 48, 'se': None, 'stride': 1, 'heads': 2},
#         # stage4
#         {'inp': 48, 'exp': 288, 'out': 96, 'se': None, 'stride': 2, 'heads': 2},
#         {'inp': 96, 'exp': 384, 'out': 96, 'se': None, 'stride': 1, 'heads': 2},
#         {'inp': 96, 'exp': 576, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
#         {'inp': 128, 'exp': 768, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
#         # stage5
#         {'inp': 128, 'exp': 768, 'out': 192, 'se': None, 'stride': 2, 'heads': 2},
#         {'inp': 192, 'exp': 1152, 'out': 192, 'se': None, 'stride': 1, 'heads': 2},
#         {'inp': 192, 'exp': 1152, 'out': 192, 'se': None, 'stride': 1, 'heads': 2},
#     ],
#     'fc1': 1920,  # hid_layer
#     'fc2': 150  # num_clasess
#     ,
# }

class MobileFormer(nn.Module):
    def __init__(self, cfg):
        super(MobileFormer, self).__init__()
        self.token = nn.Parameter(nn.Parameter(torch.randn(1, cfg['token'], cfg['embed'])))
        # stem 3 224 224 -> 16 112 112
        # self.stem = nn.Sequential(
        #     nn.Conv3d(3, cfg['stem'], kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm3d(cfg['stem']),
        #     hswish(),
        # )
        # bneck
        self.stem = nn.Sequential(
            nn.Conv3d(3, cfg['stem'], kernel_size=3, stride=2, padding=1,bias=False),
            nn.BatchNorm3d(cfg['stem']),
            hswish()
        )
        # stem 16
        self.bneck = nn.Sequential(
            nn.Conv3d(cfg['stem'], cfg['bneck']['e'],  kernel_size=3, stride=cfg['bneck']['s'], padding=1, groups=cfg['stem']),
            hswish(),
            nn.Conv3d(cfg['bneck']['e'], cfg['bneck']['o'], kernel_size=1, stride=1),
            nn.BatchNorm3d(cfg['bneck']['o'])
        )
        # self.bneck = nn.Sequential(
        #     nn.Conv3d(cfg['stem'], cfg['bneck']['e'], 3, stride=cfg['bneck']['s'], padding=1, groups=cfg['stem']),
        #     hswish(),
        #     nn.Conv3d(cfg['bneck']['e'], cfg['bneck']['o'], kernel_size=1, stride=1),
        #     nn.BatchNorm3d(cfg['bneck']['o'])
        # )

        # body
        self.block = nn.ModuleList()
        for kwargs in cfg['body']:
            self.block.append(BaseBlock(**kwargs, dim=cfg['embed']))
        inp = cfg['body'][-1]['out']
        exp = cfg['body'][-1]['exp']
        self.conv = nn.Conv3d(inp, exp, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm3d(exp)
        self.avg = nn.AvgPool3d((1, 7, 7))
        self.head = nn.Sequential(
            nn.Linear(exp + cfg['embed'], cfg['fc1']),
            nn.BatchNorm1d(cfg['fc1']),
            hswish(),
            nn.Linear(cfg['fc1'], cfg['fc2'])
        )
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.transpose(1,2)
        b,c,t,h,w = x.shape
        z = self.token.repeat(b, 1, 1)
        # pdb.set_trace()
        x = self.bneck(self.stem(x))
        # print("1 z x",torch.max(z),torch.max(x))
        # print("2",torch.sum(z),torch.sum(x))
        idx =0
        for m in self.block:
            # if torch.isnan(m([x, z])[0]).sum() != 0:
            # if torch.max( m([x, z])[0])>1e3 or torch.max( m([x, z])[1])>=1e3:
            #     pdb.set_trace()
            x, z = m([x, z])
            idx+=1
            # print("m",idx,torch.sum(z),torch.sum(x))
        # print("2 z x",torch.max(z),torch.max(x))
        # x, z = self.block([x, z])
        x = self.avg(self.bn(self.conv(x))).view(b, -1) # [8, 1152]
        # pdb.set_trace()
        z = z[:, 0, :].view(b, -1) # [8, 192]
        # z = z.view(b,-1)
        out = torch.cat((x, z), -1)
        # print("2",torch.sum(self.head(out)))
        return self.head(out)
        # return x, z


if __name__ == "__main__":
    model = MobileFormer(config_52)
    inputs = torch.randn((3, 3, 224, 224))
    print(inputs.shape)
    # for i in range(100):
    #     t = time.time()
    #     output = model(inputs)
    #     print(time.time() - t)
    print("Total number of parameters in networks is {} M".format(sum(x.numel() for x in model.parameters()) / 1e6))
    output = model(inputs)
    print(output.shape)
