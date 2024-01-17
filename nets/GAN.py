import torch.nn as nn
import torch

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_channels//2,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=9):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # Question here
        in_channels = 64
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)
        self.up4 = UpBlock(in_channels*16, in_channels*4, nb_Conv=2)
        self.up3 = UpBlock(in_channels*8, in_channels*2, nb_Conv=2)
        self.up2 = UpBlock(in_channels*4, in_channels, nb_Conv=2)
        self.up1 = UpBlock(in_channels*2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1))
        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None
    def forward(self, x):
        # Question here
        x = x.float()
        # print(x.shape)
        x1 = self.inc(x)
        print("x1",x1.shape)
        x2 = self.down1(x1)
        print("x2", x2.shape)
        x3 = self.down2(x2)
        print("x3", x3.shape)
        x4 = self.down3(x3)
        print("x4", x4.shape)
        x5 = self.down4(x4)
        print("x5", x5.shape)
        x = self.up4(x5, x4)
        print("up4", x.shape)
        x = self.up3(x, x3)
        print("up3", x.shape)
        x = self.up2(x, x2)
        print("up2", x.shape)
        x = self.up1(x, x1)
        print("up1", x.shape)
        exit()
        if self.last_activation is not None:
            logits = self.last_activation(self.outc(x))
            # print("111")
        else:
            logits = self.outc(x)
            # print("222")
        # logits = self.outc(x) # if using BCEWithLogitsLoss
        # print(logits.size())
        return logits


class ConvBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, kernel_size=3, stride=2, padding=1, activation='lrelu'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_num_ch, out_num_ch, kernel_size, stride, padding=padding),
            nn.BatchNorm2d(out_num_ch)
            )
        if activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        if activation == 'elu':
            self.act = nn.ELU(inplace=True)
        else:
            self.act = nn.Sequential()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class Discriminator(nn.Module):
    def __init__(self,in_channel=1,out_channel=1, base_n_filter = 8,training=True):
        super(Discriminator, self).__init__()
        self.training = training
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = nn.Conv2d(in_channel,base_n_filter*4,kernel_size=(3,3))
        self.batchnorm1 = nn.BatchNorm2d(base_n_filter*4)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.avgpool1 = nn.AvgPool2d(kernel_size=(2,2),stride=(2,2))

        self.conv2 = nn.Conv2d(base_n_filter*4,base_n_filter*8,kernel_size=(3,3))
        self.batchnorm2 = nn.BatchNorm2d(base_n_filter*8)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.avgpool2 = nn.AvgPool2d(kernel_size=(2,2),stride=(2,2))

        self.conv3 = nn.Conv2d(base_n_filter*8,base_n_filter*16,kernel_size=(3,3))
        self.batchnorm3 = nn.BatchNorm2d(base_n_filter*16)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.avgpool3 = nn.AvgPool2d(kernel_size=(2,2),stride=(2,2))

        self.conv4 = nn.Conv2d(base_n_filter * 16, base_n_filter * 32, kernel_size=(3, 3))
        self.batchnorm4 = nn.BatchNorm2d(base_n_filter * 32)
        self.relu4 = nn.LeakyReLU(inplace=True)
        self.avgpool4 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv5 = nn.Conv2d(base_n_filter * 32, base_n_filter * 64, kernel_size=(3, 3))
        self.batchnorm5 = nn.BatchNorm2d(base_n_filter * 64)
        self.relu5 = nn.LeakyReLU(inplace=True)
        self.avgpool5 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        #
        self.conv6 = nn.Conv2d(base_n_filter * 64, base_n_filter * 128, kernel_size=(3, 3))
        self.batchnorm6 = nn.BatchNorm2d(base_n_filter * 128)
        self.relu6 = nn.LeakyReLU(inplace=True)
        self.avgpool6 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        #
        self.conv7 = nn.Conv2d(base_n_filter * 128, base_n_filter * 256, kernel_size=(3, 3))
        self.batchnorm7 = nn.BatchNorm2d(base_n_filter * 256)
        self.relu7 = nn.LeakyReLU(inplace=True)
        self.avgpool7 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))


        self.liner = nn.Linear(1280000,base_n_filter*256)    #线性值要改
        self.drop = nn.Dropout(p=0.6, inplace= False)

        self.liner2 = nn.Linear(base_n_filter*256,out_channel)
        self.activate_function = nn.Sigmoid()

    def forward(self, x):
        # print("xxxxxxx",x.shape)
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu1(out)
        # out = self.avgpool1(out)

        out = self.conv2(out)

        out = self.batchnorm2(out)
        out = self.relu2(out)
        # out = self.avgpool2(out)

        out = self.conv3(out)
        out = self.batchnorm3(out)
        out = self.relu3(out)
        # out = self.avgpool3(out)
        # print(out.shape)

        out = self.conv4(out)
        out = self.batchnorm4(out)
        out = self.relu4(out)
        # out = self.avgpool4(out)

        out = self.conv5(out)
        out = self.batchnorm5(out)
        out = self.relu5(out)
        out = self.avgpool5(out)

        out = self.conv6(out)
        out = self.batchnorm6(out)
        out = self.relu6(out)
        out = self.avgpool6(out)

        #
        out = self.conv7(out)
        out = self.batchnorm7(out)
        out = self.relu7(out)
        out = self.avgpool7(out)
        # print(out.shape)

        out = out.view(out.size(0), -1)

        out = self.liner(out)
        out = self.drop(out)

        out = self.liner2(out)
        out = self.activate_function(out)
        return out


