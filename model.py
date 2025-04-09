import torch as torch
import torch.nn as nn
from torchvision import models

class AttentionGate(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )

        self.Ws = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )

        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(out_c, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        out = self.relu(Wg + Ws)
        out = self.output(out)
        return out * g
    

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DoubleConv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, up_in_channels, x_in_channels, kernel_size=3, padding=1, dropout=0):
        super(Decoder, self).__init__()

        self.upsample = nn.ConvTranspose2d(up_in_channels, up_in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)

        in_channels = up_in_channels // 2 + x_in_channels
        out_channels = in_channels // 2

        self.layers = DoubleConv(in_channels, out_channels, kernel_size, padding)

    def forward(self, skip_connection, x):
        x = self.upsample(x)
        x = AttentionGate([x.shape[1], skip_connection.shape[1]], x.shape[1])(x, skip_connection)
        return self.layers(torch.cat([skip_connection, x], dim=1))


class AttenResUnet(nn.Module):
    def __init__(self):

        backbone = models.resnet34(pretrained=True)

        super(AttenResUnet, self).__init__()

        for param in backbone.parameters():
            param.requires_grad = False

        self.encoder0 = nn.Sequential(
                        backbone.conv1, 
                        backbone.bn1, 
                        backbone.relu
                        )
        
        self.maxpool = backbone.maxpool

        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.middle = DoubleConv(512, 512)

        self.decoder0 = Decoder(512, 256)
        self.decoder1 = Decoder(256, 128)
        self.decoder2 = Decoder(128, 64)
        self.decoder3 = Decoder(64, 64)

        self.resize = nn.ConvTranspose2d(48, 16, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.final_layer = nn.Sequential(
            nn.Conv2d(16, 21, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(21),
            nn.ReLU(inplace=True),
            )


    def forward(self, x):
        a1 = self.encoder0(x)
        a1_pooled = self.maxpool(a1)
        a2 = self.encoder1(a1_pooled)
        a3 = self.encoder2(a2)
        a4 = self.encoder3(a3)
        a5 = self.encoder4(a4)

        mid = self.middle(a5)

        d1 = self.decoder0(a4, mid)
        d2 = self.decoder1(a3, d1)
        d3 = self.decoder2(a2, d2)
        d4 = self.decoder3(a1, d3)

        resized = self.resize(d4)
        final = self.final_layer(resized)

        return final
