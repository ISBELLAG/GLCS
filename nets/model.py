import torch
from torch import nn
from nets.efficientseg import BiFPN_1toN_semi, Segmentor, EfficientNet


class EfficientSegBackbone_PRM_semi(nn.Module):
    def __init__(self, num_classes=80, compound_coef=0, load_weights=False, **kwargs):
        super(EfficientSegBackbone_PRM_semi, self).__init__()
        self.compound_coef = compound_coef

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        conv_channel_coef = {
            # the channels of P1/P2/P3/P4/P5.
            0: [16, 24, 40, 112, 320],
            1: [16, 24, 40, 112, 320],
            2: [16, 24, 48, 120, 352],
            3: [24, 32, 48, 136, 384],
            4: [24, 32, 56, 160, 448],
            5: [24, 40, 64, 176, 512],
            6: [32, 40, 72, 200, 576],
            7: [32, 40, 72, 200, 576],
        }

        self.bifpn = nn.Sequential(
            *[BiFPN_1toN_semi(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.num_classes = num_classes
        self.segmentor = Segmentor(in_channels=self.fpn_num_filters[self.compound_coef], num_classes=num_classes, num_layers=self.box_class_repeats[self.compound_coef])

        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs, mode):#, labels_shows):
        max_size = inputs.shape[-1]

        p1, p2, p3, p4, p5 = self.backbone_net(inputs)

        features = (p1, p2, p3, p4, p5, mode)
        features = self.bifpn(features)

        segmentation = self.segmentor(features[0:5])

        return segmentation, features[5], features[6]

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')


class PRM(nn.Module):
    def __init__(self):
        super(PRM, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, l_p, ul_p, num_l):

        _, C, height, width = ul_p.size()

        proj_query = ul_p.view(num_l, -1, width * height).permute(0, 2, 1)
        proj_key = l_p.view(num_l, -1, width * height)

        energy = torch.bmm(proj_query / proj_key.max(), proj_key / proj_key.max())
        energy = energy.to(torch.float32) * proj_key.max() * proj_key.max()

        # energy = torch.bmm(proj_query, proj_key)

        attention = self.softmax(energy)
        proj_value = l_p.view(num_l, -1, width * height)

        attentionM = torch.bmm(proj_value, attention.permute(0, 2, 1))
        attentionM = attentionM.view(ul_p.size()[0], C, height, width)

        return attentionM

